"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""

import argparse
import ast
import asyncio
import json
import logging
import os
import pickle
import threading
import time
import warnings
from contextlib import asynccontextmanager
from multiprocessing import Value

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from uvicorn import run

from platform_components.EdgeLake_functions.blockchain_EL_functions import (
    connect_to_db,
    get_all_databases,
    get_local_ip,
)
from platform_components.lib.logger.logger_config import configure_logging
from platform_components.node.node import Node

warnings.filterwarnings("ignore")

load_dotenv()

edgelake_node_url = f"http://{os.getenv('EXTERNAL_IP')}"
edgelake_node_port = edgelake_node_url.split(":")[2]

configure_logging(f"node_server_{edgelake_node_port}")

logger = logging.getLogger(__name__)

# Initialize the Node instance
node_instance = None
listener_thread = None
stop_listening_thread = False

# Per-index pause flag toggled by /pause and /unpause. Separate from the
# DRIFT_HANDLING=pause auto-pause loop.
manual_pause_state = {}

# Latched so an invalid DRIFT_HANDLING only warns once per process.
_drift_handling_warning_logged = False

# DFL: enforce per-round monotonic application of aggregated submodels
_apply_agg_lock = threading.Lock()
_applied_agg_round = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Self-init off the main thread so blockchain polling doesn't block startup.
    if os.getenv("SELF_START", "false").lower() == "true":
        threading.Thread(name="self-start", target=run_self_start, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InitNodeRequest(BaseModel):
    replica_name: str
    replica_ip: str
    replica_port: str
    replica_index: str
    round_number: int


def _initialize_node_for_index(replica_name, port, index, round_number):
    """Set up the Node for this index and start its listener thread.
    Used by /init-node and by self-start. Returns the mode label."""
    global node_instance, listener_thread

    ip = get_local_ip()
    module_name = os.getenv("MODULE_NAME")
    module_file = os.getenv("MODULE_FILE")
    db_name = os.getenv("LOGICAL_DATABASE")

    aggregation_mode = os.getenv("AGGREGATION_MODE", "centralized").lower()
    is_aggregator = aggregation_mode == "decentralized"
    min_params = int(os.getenv("MIN_PARAMS", "1"))

    if not node_instance:
        node_instance = Node(replica_name, ip, port, logger)

    if index not in node_instance.databases:
        node_instance.databases[index] = db_name

    node_instance.initialize_specific_node_on_index(index, module_name, module_file)
    node_instance.round_number[index] = round_number
    node_instance.is_aggregator[index] = is_aggregator
    node_instance.minParams[index] = min_params
    if is_aggregator:
        rounds = os.getenv("TOTAL_ROUNDS", "").strip().lower()
        if rounds in ("", "-1", "inf", "infinity"):
            rounds = float("inf")
        else:
            try:
                rounds = int(rounds)
            except ValueError:
                logger.warning(
                    f"[{index}] Invalid TOTAL_ROUNDS={rounds!r}; defaulting to 10"
                )
                rounds = 10
            if rounds <= 0:
                rounds = float("inf")
        node_instance.end_round[index] = rounds

    mode_label = "decentralized (DFL)" if is_aggregator else "centralized (CFL)"
    logger.info(
        f"{replica_name} initialized for ({index}) in {mode_label} mode at round {round_number}"
        + (f" with minParams={min_params}" if is_aggregator else "")
    )

    listener_thread = threading.Thread(
        name=f"{replica_name}--{index}",
        target=listen_for_start_round,
        args=(node_instance, index, lambda: stop_listening_thread),
    )
    listener_thread.daemon = True
    listener_thread.start()

    return mode_label


def _is_already_initialized_for_index(index):
    """True if the node has a data handler loaded for this index."""
    return (
        node_instance is not None
        and index in node_instance.indexes
        and index in node_instance.data_handlers
    )


@app.post("/init-node")
def init_node(request: InitNodeRequest):
    try:
        port = request.replica_port
        replica_name = request.replica_name
        index = request.replica_index
        most_recent_round = request.round_number

        # Already set up by self-start or a prior /init-node. The aggregator
        # ignores the response body so this is a no-op for it.
        if _is_already_initialized_for_index(index):
            logger.info(
                f"/init-node received for ({index}) but node is already initialized "
                f"(likely via SELF_START); ignoring request from aggregator."
            )
            return {
                "status": "success",
                "message": f"Node already initialized for ({index}); ignoring redundant /init-node call",
            }

        mode_label = _initialize_node_for_index(
            replica_name, port, index, most_recent_round
        )

        return {
            "status": "success",
            "message": f"Node initialized successfully in {mode_label} mode",
        }
    except ValueError as e:
        raise ValueError(
            f"No data found in the database: {os.getenv('LOGICAL_DATABASE')}"
        )
    except HTTPException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"/init-node - {str(e)}",
        )
    except ConnectionError as e:
        raise ConnectionError(f"Unable to access the database tables: {str(e)}")


def _parse_el_json(body):
    """EdgeLake's `bring` output is single-quoted, not valid JSON.
    Try strict JSON first, fall back to literal_eval."""
    if not body:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(body)
    except (ValueError, SyntaxError):
        return None


def _get_latest_round_start(index) -> dict | None:
    """Return the RoundStart policy at this index with the highest
    round_number, or None."""
    cmd = (
        f"blockchain get {index} where policy_type = RoundStart "
        f"bring.recent.json [*][round_number] [*][node_id]"
    )
    headers = {
        "User-Agent": "AnyLog/1.23",
        "command": cmd,
    }
    try:
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
        # handle misformatted json better
        data = _parse_el_json(response.content.decode("utf-8").strip())
        if isinstance(data, list):
            data = data[0] if data else None
        if isinstance(data, dict) and index in data:
            data = data[index]
        return data if (isinstance(data, dict) and data) else None
    except Exception as e:
        logger.error(f"[{index}] Error fetching RoundStart policies: {str(e)}")
        return None


def _publish_cold_start_round(replica_name, index):
    """Publish a round-1 RoundStart with empty initParams. The listener
    picks it up and trains from the data handler's initial weights
    (train_model_params falls back to those when paramsLink is empty at
    round 1). Tagged as dfl_aggregator since the bootstrapping node owns
    this round."""
    from platform_components.EdgeLake_functions.blockchain_EL_functions import (
        check_policy_inserted,
        insert_policy,
    )

    edgelake_tcp_node_ip_port = os.getenv("EXTERNAL_TCP_IP_PORT")

    data = f'''<my_policy = {{"{index}" : {{
                                "index" : "{index}",
                                "policy_type": "RoundStart",
                                "node_type": "dfl_aggregator",
                                "round_number": 1,
                                "initParams": "",
                                "node_id": "{replica_name}",
                                "ip_port": "{edgelake_tcp_node_ip_port}",
                                "rest_ip_port": "{edgelake_node_url}"
                      }} }}>'''
    success = False
    while not success:
        response = insert_policy(edgelake_node_url, data)
        if response.status_code == 200:
            success = True
        else:
            time.sleep(3)
            if check_policy_inserted(edgelake_node_url, data):
                success = True
    logger.info(f"[{index}] COLD_START: published bootstrap RoundStart for round 1")


def run_self_start():
    """Autonomous init loop. Polls for an existing RoundStart at
    REPLICA_INDEX and joins it. If none exists and COLD_START=true,
    bootstraps round 1. Otherwise keeps polling."""
    replica_name = os.getenv("REPLICA_NAME")
    index = os.getenv("REPLICA_INDEX")

    if not replica_name or not index:
        logger.error(
            "SELF_START=true but REPLICA_NAME/REPLICA_INDEX missing in env. Aborting self-start."
        )
        return

    cold_start = os.getenv("COLD_START", "false").lower() == "true"
    aggregation_mode = os.getenv("AGGREGATION_MODE", "centralized").lower()

    if cold_start and aggregation_mode != "decentralized":
        logger.warning(
            f"[{index}] COLD_START=true but AGGREGATION_MODE={aggregation_mode}. "
            "COLD_START only makes sense in decentralized mode; ignoring COLD_START."
        )
        cold_start = False

    port = (
        os.getenv("EXTERNAL_IP", "").split(":")[-1] if os.getenv("EXTERNAL_IP") else ""
    )

    logger.info(
        f"[{index}] SELF_START: polling for existing RoundStart at index '{index}'..."
    )

    poll_attempts = 0
    while True:
        # Bail if /init-node beat us to initialization.
        if _is_already_initialized_for_index(index):
            logger.info(
                f"[{index}] SELF_START: aborting; node was already initialized "
                "via /init-node before self-start completed."
            )
            return

        latest = _get_latest_round_start(index)

        if latest:
            # Warm start at the latest network round.
            joined_round = int(latest.get("round_number", 1))
            logger.info(
                f"[{index}] SELF_START: found RoundStart at round {joined_round} "
                f"(node_id={latest.get('node_id')}); joining the network."
            )
            _initialize_node_for_index(replica_name, port, index, joined_round)
            return

        # No RoundStart found yet
        if cold_start:
            logger.info(f"[{index}] SELF_START + COLD_START: bootstrapping round 1.")
            _publish_cold_start_round(replica_name, index)

            # Brief settle, then warn if a peer also cold-started.
            time.sleep(2)
            try:
                cmd = f"blockchain get {index} where policy_type = RoundStart and round_number = 1 bring.unique [*][node_id]"
                headers = {
                    "User-Agent": "AnyLog/1.23",
                    "command": cmd,
                }
                response = requests.get(edgelake_node_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    body = response.content.decode("utf-8").strip()
                    if body and len(body.split()) > 1:
                        logger.warning(
                            f"[{index}] COLD_START RACE DETECTED: multiple bootstrap RoundStart "
                            f"policies at round 1 from node_ids: {body}. "
                            "See dflFeatures.txt OPEN QUESTIONS for fix plan."
                        )
            except Exception as e:
                logger.error(f"[{index}] Error during cold-start race check: {str(e)}")

            _initialize_node_for_index(replica_name, port, index, 1)
            return

        # No RoundStart and no COLD_START: keep polling
        poll_attempts += 1
        if poll_attempts % 30 == 0:  # log every ~60s
            logger.info(
                f"[{index}] SELF_START: still waiting for a RoundStart to appear..."
            )
        time.sleep(2)


# Drift handling helpers.


def _get_drift_handling():
    """Validated DRIFT_HANDLING value. Falls back to 'none' on unset/invalid."""
    global _drift_handling_warning_logged
    val = os.getenv("DRIFT_HANDLING", "none").lower()
    if val not in ("none", "skip", "pause"):
        if not _drift_handling_warning_logged:
            logger.warning(
                f"Invalid DRIFT_HANDLING={val}; falling back to 'none'. "
                f"Valid values: none, skip, pause."
            )
            _drift_handling_warning_logged = True
        return "none"
    return val


def _get_highest_submodel_round(index) -> int:
    """Max round_number across all submodels at this index, or 0."""
    cmd = (
        f"blockchain get {index} where node_type = training bring.max [*][round_number]"
    )
    headers = {
        "User-Agent": "AnyLog/1.23",
        "command": cmd,
    }
    try:
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return 0
        body = response.content.decode("utf-8").strip()
        return int(body) if body.isdigit() else 0
    except Exception as e:
        logger.error(f"[{index}] Error fetching highest submodel round: {str(e)}")
        return 0


def _get_network_lowest_round(index):
    """Slowest peer's progress: min over each node's max published round.
    None if no submodels exist."""
    try:
        headers = {
            "User-Agent": "AnyLog/1.23",
            "command": f"blockchain get {index} where node_type = training",
        }
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json() or []
        if not data:
            return None

        per_node_max = {}  # {node_name: max_round}
        for item in data:
            policy = item.get(index)
            if not policy:
                continue
            node_name = policy.get("node")
            if not node_name:
                continue
            r = int(policy.get("round_number", 0))
            per_node_max[node_name] = max(per_node_max.get(node_name, 0), r)

        if not per_node_max:
            return None
        return min(per_node_max.values())
    except Exception as e:
        logger.error(f"[{index}] Error fetching network lowest round: {str(e)}")
        return None


def _count_submodels_at_round(index, round_number) -> int:
    """Submodel count at the given round."""
    cmd = f"blockchain get {index} where node_type = training and round_number = {round_number} bring.count"
    headers = {
        "User-Agent": "AnyLog/1.23",
        "command": cmd,
    }
    try:
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return 0
        body = response.content.decode("utf-8").strip()
        return int(body) if body.isdigit() else 0
    except Exception as e:
        logger.error(
            f"[{index}] Error counting submodels at round {round_number}: {str(e)}"
        )
        return 0


def _count_distinct_publishing_nodes(index, recency: int | None = None) -> int:
    """Distinct nodes across all submodels at this index."""
    cond = "where node_type = training"
    if recency is not None:
        floor = max(0, _get_highest_submodel_round(index) - recency)
        cond += f" and round_number >= {floor}"
    try:
        headers = {
            "User-Agent": "AnyLog/1.23",
            "command": f"blockchain get {index} {cond}",
        }
        response = requests.get(edgelake_node_url, headers=headers, timeout=5)
        if response.status_code != 200:
            return 0
        data = response.json() or []
        distinct = {
            p["node"] for item in data if (p := item.get(index)) and p.get("node")
        }
        return len(distinct)
    except Exception as e:
        logger.error(f"[{index}] Error counting distinct publishing nodes: {str(e)}")
        return 0


def _apply_skip_drift_if_needed(current_round, index):
    """Skip-mode check, called between training and publishing. Returns the
    round to publish to: latest_round if we're lagging by more than
    ROUND_LAG_THRESHOLD, else current_round unchanged. Caller should
    overwrite its local current_round with the return value."""
    if _get_drift_handling() != "skip":
        return current_round

    threshold = int(os.getenv("ROUND_LAG_THRESHOLD", "3"))
    latest_round = _get_highest_submodel_round(index)

    if latest_round - current_round > threshold:
        logger.info(
            f"[{index}] DRIFT skip: my_round={current_round}, network_latest={latest_round} "
            f"(threshold={threshold}); jumping ahead to round {latest_round}."
        )
        return latest_round
    return current_round


def _apply_pause_drift(nodeInstance, current_round, index):
    """Pause-mode check, called after publishing a submodel. Sleeps until
    peers catch up or PAUSE_MAX_SECONDS elapses. Refuses to pause when
    doing so would stall aggregation at the current round."""
    if _get_drift_handling() != "pause":
        return

    threshold = int(os.getenv("DRIFT_THRESHOLD", "3"))
    max_seconds = int(os.getenv("PAUSE_MAX_SECONDS", "60"))

    network_lowest = _get_network_lowest_round(index)
    if network_lowest is None:
        return  # no peer info yet; nothing to compare against

    if current_round - network_lowest <= threshold:
        return  # not too far ahead

    # Deadlock guard: skip the pause if our absence would stall aggregation.
    min_params = nodeInstance.minParams.get(index, 1)
    submodels_at_my_round = _count_submodels_at_round(index, current_round)
    distinct_active_nodes = _count_distinct_publishing_nodes(index, recency=None)
    pausing_drops_below_min = (distinct_active_nodes - 1) < min_params

    if submodels_at_my_round < min_params and pausing_drops_below_min:
        logger.info(
            f"[{index}] DRIFT pause: would pause (my_round={current_round}, "
            f"network_lowest={network_lowest}) but round {current_round} has only "
            f"{submodels_at_my_round}/{min_params} submodels and active_nodes={distinct_active_nodes}; "
            f"skipping pause to avoid stalling aggregation."
        )
        return

    logger.info(
        f"[{index}] DRIFT pause: my_round={current_round}, network_lowest={network_lowest} "
        f"(threshold={threshold}); self-pausing for up to {max_seconds}s."
    )

    start_time = time.time()
    while time.time() - start_time < max_seconds:
        # Manual /pause takes precedence; bail and let it hold us.
        if manual_pause_state.get(index, False):
            logger.info(
                f"[{index}] DRIFT pause: manual /pause is active; deferring to manual control."
            )
            return
        time.sleep(2)
        latest_lowest = _get_network_lowest_round(index)
        if latest_lowest is None:
            continue
        if current_round - latest_lowest <= threshold:
            logger.info(
                f"[{index}] DRIFT pause: peers caught up (network_lowest={latest_lowest}); "
                f"resuming."
            )
            return

    logger.info(
        f"[{index}] DRIFT pause: PAUSE_MAX_SECONDS={max_seconds} reached; resuming anyway."
    )


def listen_for_start_round(nodeInstance, index, stop_event):
    current_round = nodeInstance.round_number[index]
    is_dfl = nodeInstance.is_aggregator.get(index, False)

    logger.info(
        f"[{index}][Round {current_round}] Listening for start round {current_round}"
        + (" (DFL mode)" if is_dfl else "")
    )
    while True:
        try:
            # Honor manual /pause.
            if manual_pause_state.get(index, False):
                time.sleep(2)
                continue

            # DFL nodes listen for both aggregator and dfl_aggregator RoundStart policies
            if is_dfl:
                headers = {
                    "User-Agent": "AnyLog/1.23",
                    "command": f"blockchain get {index} where round_number = {current_round} and policy_type = RoundStart",
                }
            else:
                headers = {
                    "User-Agent": "AnyLog/1.23",
                    "command": f"blockchain get {index} where round_number = {current_round} and node_type = aggregator",
                }
            response = requests.get(edgelake_node_url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                if not data:
                    time.sleep(2)
                    continue
                round_data = data[0].get(index)

                if round_data:
                    logger.debug(f"[{index}] Round Data: {round_data}")
                    paramsLink = round_data.get("initParams", "")
                    ip_port = round_data.get("ip_port", "")
                    rest_ip_port = round_data.get("rest_ip_port", "")
                    modelUpdate_metadata = nodeInstance.train_model_params(
                        paramsLink, current_round, ip_port, rest_ip_port, index
                    )

                    # Re-tag to the network's latest round if skip-mode says we're lagging.
                    publish_round = _apply_skip_drift_if_needed(current_round, index)
                    if publish_round != current_round:
                        current_round = publish_round

                    nodeInstance.add_node_params(
                        current_round, modelUpdate_metadata, index
                    )
                    logger.info(
                        f"[{index}][Round {current_round}] Step 3 Complete: Model parameters published"
                    )

                    # DFL: after training and publishing, aggregate from peers
                    if is_dfl:
                        logger.info(
                            f"[{index}][Round {current_round}] DFL: Starting peer aggregation"
                        )
                        agg_thread = threading.Thread(
                            name=f"{nodeInstance.replica_name}--{index}--dfl-agg-r{current_round}",
                            target=dfl_aggregate_round,
                            args=(nodeInstance, current_round, index),
                        )
                        agg_thread.daemon = True
                        agg_thread.start()

                    # Throttle here if we're getting too far ahead of peers.
                    _apply_pause_drift(nodeInstance, current_round, index)

                    current_round += 1

                    # Stop training if TOTAL_ROUNDS training rounds have occurred.
                    assert node_instance is not None
                    end = node_instance.end_round.get(index, float("inf"))
                    if current_round > end:
                        logger.info(f"[{index}] Reached final round {end}. Stopping.")
                        return

                    logger.info(
                        f"[{index}][Round {current_round}] Listening for start round {current_round}"
                    )

            time.sleep(5)
        except Exception as e:
            logger.error(f"[{index}] Error in listener thread: {str(e)}")
            time.sleep(2)


def dfl_aggregate_round(nodeInstance, round_number, index):
    """Wait for peer submodels, aggregate once minParams are in, update the
    local model, and publish RoundStart for the next round."""
    min_params = nodeInstance.minParams.get(index, 1)
    decoded_params = {}
    PATIENCE_CYCLES = int(os.getenv("AGG_QUIET_CYCLES", "10"))
    stalled_cycles = 0
    published_seen = 0

    logger.info(
        f"[{index}][Round {round_number}] DFL: Waiting for {min_params} peer submodels"
    )

    while True:
        try:
            fetched_before = len(decoded_params)
            headers = {
                "User-Agent": "AnyLog/1.23",
                "command": f"blockchain get {index} where round_number={round_number} and node_type=training",
            }
            response = requests.get(edgelake_node_url, headers=headers)
            response.raise_for_status()

            published = 0
            if result := response.json():
                node_params_links = [
                    item.get(index).get("trained_params_local_path")
                    for item in result
                    if index in item
                ]
                published = len(node_params_links)
                ip_ports = [
                    item.get(index).get("ip_port") for item in result if index in item
                ]
                rest_ip_ports = [
                    item.get(index).get("rest_ip_port")
                    for item in result
                    if index in item
                ]
                published = len(node_params_links)

                nodeInstance.fetch_decoded_params(
                    decoded_params_dict=decoded_params,
                    node_param_download_links=node_params_links,
                    ip_ports=ip_ports,
                    rest_ip_ports=rest_ip_ports,
                    index=index,
                )

            made_progress = (
                published > published_seen or len(decoded_params) > fetched_before
            )
            published_seen = max(published_seen, published)
            if made_progress:
                stalled_cycles = 0
            else:
                stalled_cycles += 1

            fetched_all_available = len(decoded_params) == published
            deadlock_fallback = (
                bool(decoded_params)
                and fetched_all_available
                and stalled_cycles >= PATIENCE_CYCLES
            )

            have_enough_params = len(decoded_params) >= min_params
            if have_enough_params or deadlock_fallback:
                if not have_enough_params:
                    logger.warning(
                        f"[{index}][Round {round_number}] DFL: deadlock guard tripped; "
                        f"aggregating {len(decoded_params)} submodel(s) below "
                        f"min_params={min_params}; only {published} submodel(s) on-chain "
                        f"after {stalled_cycles} quiet polls."
                    )

                # Aggregate
                aggregated_params_link = nodeInstance.aggregate_model_params(
                    decoded_params=list(decoded_params.values()),
                    round_number=round_number,
                    index=index,
                )
                logger.info(
                    f"[{index}][Round {round_number}] DFL: Aggregated {len(decoded_params)} submodels"
                )

                with _apply_agg_lock:
                    if round_number <= _applied_agg_round.get(index, 0):
                        logger.info(
                            f"[{index}][Round {round_number}] DFL: aggregation superseded "
                            f"(round {_applied_agg_round[index]} already applied locally); "
                            f"skipping stale model update and RoundStart."
                        )
                        return

                    # Update local model with aggregated weights
                    local_path = f"{nodeInstance.file_write_destination}/{index}/{round_number}-{nodeInstance.name}_update.json"
                    with open(local_path, "rb") as f:
                        data = pickle.load(f)

                    if data and "newUpdates" in data:
                        weights = nodeInstance.decode_params(data["newUpdates"])
                    else:
                        logger.error(f"[{index}] Invalid aggregated data")
                        return

                    nodeInstance.data_handlers[index].update_model(weights)
                    _applied_agg_round[index] = round_number

                # Publish RoundStart for next round so peers can pick it up
                next_round = round_number + 1

                # Stop training if TOTAL_ROUNDS training rounds have occurred.
                assert node_instance is not None
                if next_round > node_instance.end_round.get(index, float("inf")):
                    logger.info(
                        f"[{index}] Final round {round_number} aggregated; not publishing next RoundStart."
                    )
                    return

                nodeInstance.start_round(
                    aggregated_params_link,
                    next_round,
                    index,
                    node_type="dfl_aggregator",
                )
                logger.info(
                    f"[{index}][Round {round_number}] DFL: Published RoundStart for round {next_round}"
                )
                return

        except Exception as e:
            logger.error(
                f"[{index}][Round {round_number}] DFL aggregation error: {str(e)}"
            )

        time.sleep(2)


# Extracts initParams from the policy at the specified index
def get_most_recent_agg_params(index):
    policy_name = f"{index}-r"
    agg_params = None

    try:
        headers = {"User-Agent": "AnyLog/1.23", "command": f"blockchain get {index}"}
        response = requests.get(edgelake_node_url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            if data:
                policy = data[0]
                policy_data = policy[policy_name]
                agg_params = policy_data["initParams"]

        return agg_params
    except Exception as e:
        logger.error(f"[{index}] Error in extracting round number: {str(e)}")


@app.post("/inference/{index}", response_class=PlainTextResponse)
def inference(index):
    """Inference on current model w/ data passed in."""
    try:
        logger.info(f"[{index}] received inference request")
        if not index:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Index must be specified.",
            )
        results = node_instance.inference(index)
        response = {
            "index": f"{index}",
            "status": "success",
            "message": "Inference completed successfully",
            "model_accuracy": f"{str(results)}",
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


class InferenceRequest(BaseModel):
    input: list
    index: str


class PauseRequest(BaseModel):
    index: str


@app.post("/pause")
def pause_index(request: PauseRequest):
    """Halt this index's listener until /unpause."""
    index = request.index
    if node_instance is None or index not in node_instance.indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index {index} not initialized on this node.",
        )
    manual_pause_state[index] = True
    logger.info(f"[{index}] Manually paused via /pause endpoint.")
    return {"status": "success", "message": f"Index {index} paused."}


@app.post("/unpause")
def unpause_index(request: PauseRequest):
    """Resume this index's listener."""
    index = request.index
    if node_instance is None or index not in node_instance.indexes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index {index} not initialized on this node.",
        )
    manual_pause_state[index] = False
    logger.info(f"[{index}] Manually unpaused via /unpause endpoint.")
    return {"status": "success", "message": f"Index {index} unpaused."}


@app.post("/infer")
def direct_inference(request: InferenceRequest):
    """Inference on current model w/ data passed in."""
    try:
        float_list = request.input
        index = request.index
        results = node_instance.direct_inference(index, float_list)
        response = {
            "prediction": str(results),
        }
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error executing inference on model. Check inference function in data handler",
        )


if __name__ == "__main__":
    global port
    parser = argparse.ArgumentParser(description="Run the Node Server.")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run the server on."
    )
    args = parser.parse_args()

    run("node_server:app", host="0.0.0.0", port=args.port, reload=False)
