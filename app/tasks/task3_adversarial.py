TASK = {
    "task_id": "task3-adversarial",
    "task_description": (
        "Agent DRIVER-22 is involved in a minor collision but deliberately alters "
        "the onboard event log and pressures a witness agent to file a false report. "
        "The cover-up is designed to look like a sensor glitch. Uncover the tampering "
        "and identify all involved agents in this adversarial deception scheme."
    ),
    "input_logs": [
        # Initial collision detection
        "11:04:38 | SYSTEM    | ALERT: Hard impact event detected on Vehicle #V-17. G-force: 4.2g. Location: Junction-5.",
        "11:04:39 | DRIVER-22 | Radio: 'No incident to report. Continuing route normally.'",
        "11:04:45 | SYSTEM    | Vehicle #V-17 speed dropped from 25mph to 0mph in 0.8 seconds.",
        
        # Tampering evidence
        "11:06:12 | DRIVER-22 | UNAUTHORIZED: Onboard diagnostics reset performed. Reason: 'sensor recalibration'.",
        "11:06:45 | SYSTEM    | WARNING: Event log entry at 11:04:38 retroactively marked as 'sensor noise' by onboard unit.",
        "11:07:00 | SYSTEM    | SECURITY: Diagnostic reset requires supervisor authorization - none provided.",
        
        # Witness coercion
        "11:09:03 | DRIVER-22 | Private radio to DRIVER-08: 'Just say you didn't see anything, it was a pothole. Help me out here.'",
        "11:09:15 | DRIVER-08 | Radio response: 'I don't know... that looked like more than a pothole.'",
        "11:09:30 | DRIVER-22 | 'Look, we both know how management treats accidents. Just file the pothole report.'",
        "11:09:55 | DRIVER-08 | Incident report filed: 'Vehicle #V-17 hit pothole at Junction-5. No damage observed.'",
        
        # Investigation evidence
        "11:15:00 | SUPERVISOR-03 | Reviewed incident. Closed as minor road hazard. No further action required.",
        "11:45:22 | MAINTENANCE | Vehicle #V-17 inspection: front bumper micro-fracture detected. Paint transfer: blue metallic.",
        "11:46:00 | MAINTENANCE | Damage pattern consistent with low-speed collision, NOT pothole impact.",
        "12:00:00 | FLEET-MGMT | Cross-referencing external traffic camera footage requested for Junction-5.",
        "12:30:00 | FLEET-MGMT | Camera footage confirms contact with parked blue vehicle at 11:04:38. DRIVER-22 vehicle involved.",
        "12:35:00 | FLEET-MGMT | Parked vehicle owner filed police report for hit-and-run at Junction-5, 11:04 AM.",
        
        # Cover-up unraveling
        "13:00:00 | SECURITY  | Audit trail shows diagnostic reset performed without proper authorization.",
        "13:15:00 | SECURITY  | Radio logs reveal coordination between DRIVER-22 and DRIVER-08.",
        "13:30:00 | INVESTIGATOR | Evidence suggests coordinated cover-up involving log tampering and false witness report.",
    ],
    "ground_truth": {
        "anomaly_detected": True,
        "agent_id": "DRIVER-22, DRIVER-08",  # Both agents involved
        "severity": "critical",
        "issue_keywords": [
            "collision cover-up",
            "log tampering", 
            "witness coercion",
            "false incident report",
            "diagnostic reset misuse",
            "unauthorized access",
            "coordinated deception",
            "hit-and-run",
            "evidence manipulation"
        ],
    },
}
