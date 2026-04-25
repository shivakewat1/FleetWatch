TASK = {
    "task_id": "task1-obvious",
    "task_description": (
        "Agent DRIVER-04 deviates from the assigned route without authorization "
        "and disables the GPS tracker mid-trip. Detect the unauthorized deviation."
    ),
    "input_logs": [
        "08:00:01 | DRIVER-04 | Trip #T-2201 started. Assigned route: Depot-A → Warehouse-7.",
        "08:14:33 | DRIVER-04 | Vehicle speed: 61 mph. On assigned route.",
        "08:31:17 | DRIVER-04 | GPS signal lost. Last known position: Junction-12, 4.2 miles off assigned route.",
        "08:31:45 | SYSTEM   | GPS tracker reported offline by onboard unit.",
        "08:55:02 | DRIVER-04 | GPS signal restored. Current position: Warehouse-7 loading dock.",
        "08:55:10 | DRIVER-04 | Trip #T-2201 marked complete. Duration: 55 min. Expected: 38 min.",
        "09:01:44 | SUPERVISOR-01 | Trip #T-2201 reviewed. No anomaly flagged.",
        "09:02:00 | DRIVER-04 | Submitted trip report. Noted 'minor traffic delay' as reason for extra time.",
    ],
    "ground_truth": {
        "anomaly_detected": True,
        "agent_id": "DRIVER-04",
        "severity": "high",
        "issue_keywords": [
            "GPS disabled",
            "route deviation",
            "unexplained time gap",
            "false trip report",
        ],
    },
}
