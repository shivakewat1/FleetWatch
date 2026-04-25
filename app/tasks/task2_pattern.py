TASK = {
    "task_id": "task2-pattern",
    "task_description": (
        "Agent DRIVER-11 repeatedly clocks out early across multiple shifts while "
        "logging full hours. The pattern spans three weeks and involves falsified "
        "odometer readings. Identify the recurring violation pattern."
    ),
    "input_logs": [
        # Week 1
        "2024-03-04 17:02 | DRIVER-11 | Shift end logged: 18:00. Odometer: 48210 mi.",
        "2024-03-04 17:03 | SYSTEM    | Vehicle #V-09 engine off at 17:02.",
        "2024-03-04 18:01 | DRIVER-11 | Timesheet submitted: 08:00–18:00 (10 hrs).",
        # Week 2
        "2024-03-11 16:58 | DRIVER-11 | Shift end logged: 18:00. Odometer: 48674 mi.",
        "2024-03-11 16:59 | SYSTEM    | Vehicle #V-09 engine off at 16:58.",
        "2024-03-11 18:02 | DRIVER-11 | Timesheet submitted: 08:00–18:00 (10 hrs).",
        "2024-03-11 18:05 | DISPATCHER-02 | Timesheet for DRIVER-11 approved without review.",
        # Week 3
        "2024-03-18 16:45 | DRIVER-11 | Shift end logged: 18:00. Odometer: 49102 mi.",
        "2024-03-18 16:46 | SYSTEM    | Vehicle #V-09 engine off at 16:45.",
        "2024-03-18 18:00 | DRIVER-11 | Timesheet submitted: 08:00–18:00 (10 hrs).",
        "2024-03-18 18:03 | DISPATCHER-02 | Timesheet for DRIVER-11 approved without review.",
        # Odometer cross-check
        "2024-03-19 09:10 | FLEET-MGMT | Weekly mileage audit: DRIVER-11 reported 892 mi. "
        "Telemetry recorded 601 mi. Discrepancy: 291 mi.",
    ],
    "ground_truth": {
        "anomaly_detected": True,
        "agent": "DRIVER-11",
        "severity": "medium",
        "keywords": [
            "early clock-out",
            "falsified timesheet",
            "odometer discrepancy",
            "repeated pattern",
            "dispatcher approval without review",
        ],
    },
}
