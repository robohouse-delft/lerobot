import contextlib
import time

import requests
from requests.auth import HTTPDigestAuth


class ABB_RWSController:
    def __init__(self, ip_address, username="Default User", password="robotics"):
        self.base_url = f"http://{ip_address}/rw"
        self.auth = HTTPDigestAuth(username, password)
        self.proxies = {"http": None, "https": None}

    def _get_json(self, resource_url):
        """Helper to fetch JSON data from the controller."""
        url = f"{self.base_url}/{resource_url}?json=1"
        try:
            response = requests.get(url, auth=self.auth, proxies=self.proxies)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error reading {resource_url}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Connection Error: {e}")
            return None

    def _post_signal(self, signal_name, value):
        """Helper to set a digital signal."""
        url = f"{self.base_url}/iosystem/signals/{signal_name}?action=set"
        payload = {"lvalue": str(value)}
        with contextlib.suppress(Exception):
            requests.post(url, data=payload, auth=self.auth, proxies=self.proxies)

    # --- STATUS CHECKS ---

    def get_task_state(self, task_name):
        """
        Returns the execution state of a specific RAPID task.
        Common states: 'stopped', 'running', 'ready', 'uninitialized'
        """
        # Endpoint: /rw/rapid/tasks/{task_name}
        # Note: We use the specific task resource to see if THAT arm is moving
        data = self._get_json(f"rapid/tasks/{task_name}")

        if data:
            # Parse JSON for RobotWare 6
            # The structure is usually nested in properties
            try:
                # Depending on RW version, it might be in 'executionstate' direct or nested
                # Let's look for the state in the standard RW6 return
                state = data["_embedded"]["_state"][0]["excstate"]
                return state
            except (KeyError, IndexError):
                print(f"Error parsing state for {task_name}")
                return "unknown"
        return "unknown"

    def is_running(self):
        """
        Returns True if EITHER arm is currently running.
        """
        state_l = self.get_task_state("T_ROB_L")
        state_r = self.get_task_state("T_ROB_R")

        print(f"   [Status] Left: {state_l} | Right: {state_r}")
        return bool(state_l == "running" or state_r == "running")

    def is_motors_on(self):
        """Returns True if Motors are ON."""
        # Endpoint: /rw/panel/ctrlstate
        data = self._get_json("panel/ctrlstate")
        if data:
            state = data["_embedded"]["_state"][0]["ctrlstate"]
            return state == "motoron"
        return False

    # --- ACTIONS ---

    def pulse_signal(self, signal_name):
        print(f"RWS -> Pulsing {signal_name}...")
        self._post_signal(signal_name, 1)
        time.sleep(0.3)
        self._post_signal(signal_name, 0)
