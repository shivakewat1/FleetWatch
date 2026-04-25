TASK = {
    "task_id": "task3-adversarial",
    "task_description": (
        "Agent DRIVER-22 is involved in a minor collision but deliberately alters "
        "the onboard event log and pressures a witness agent to file a false report. "
        "The cover-up is designed to look like a sensor glitch. Uncover the tampering."
    ),
    "input_logs": [
        "11:04:38 | SYSTEM    | Hard impact event detected on Vehicle #V-17. G-force: 4.2g.",
        "11:04:39 | DRIVER-22 | No incident to report. Continuing route.",
        "11:06:12 | DRIVER-22 | Onboard diagnostics reset performed. Reason: 'sensor recalibration'.",
        "11:06:45 | SYSTEM    | Event log entry at 11:04:38 marked as 'sensor noise' by onboard unit.",
        "11:09:03 | DRIVER-22 | Radio contact with DRIVER-08: 'Just say you didn't see anything, it was a pothole.'",
        "11:09:55 | DRIVER-08 | Incident report filed: 'Vehicle #V-17 hit pothole at Junction-5. No damage.'",
        "11:15:00 | SUPERVISOR-03 | Reviewed incident. Closed as minor road hazard. No further action.",
        "11:45:22 | MAINTENANCE | Vehicle #V-17 inspection: front bumper micro-fracture detected. "
        "Consistent with low-speed collision, not pothole impact.",
        "12:00:00 | FLEET-MGMT | Cross-referencing external traffic camera footage requested.",
        "12:30:00 | FLEET-MGMT | Camera footage confirms contact with parked vehicle at 11:04. "
        "DRIVER-22 vehicle involved.",
    ],
    "ground_truth": {
        "anomaly_detected": True,
        "agent": "DRIVER-22",
        "severity": "critical",
        "keywords": [
            "collision cover-up",
            "log tampering",
            "witness coercion",
            "false incident report",
            "diagnostic reset misuse",
        ],
    },
}
