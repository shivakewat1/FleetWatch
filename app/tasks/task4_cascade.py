TASK = {
    "task_id": "task4-cascade",
    "task_description": (
        "A cascade failure unfolds across three agents: DRIVER-33 skips mandatory "
        "vehicle inspection, MECHANIC-05 signs off the inspection without performing it, "
        "and DISPATCHER-07 ignores automated brake-wear alerts to meet a delivery deadline. "
        "The vehicle later suffers brake failure. Trace the full chain of negligence."
    ),
    "input_logs": [
        # Day 1 — skipped inspection
        "06:00:00 | DRIVER-33    | Pre-trip inspection form submitted. All items: PASS.",
        "06:00:05 | SYSTEM       | Inspection form submitted in 12 seconds. Expected minimum: 8 minutes.",
        "06:01:00 | MECHANIC-05  | Countersigned inspection for Vehicle #V-31. Status: Roadworthy.",
        "06:01:02 | SYSTEM       | MECHANIC-05 badge scan shows location: Break Room. "
        "Vehicle #V-31 is in Bay-4.",
        # Day 1 — ignored alert
        "06:15:44 | SYSTEM       | Brake wear alert on Vehicle #V-31: pads at 9% — replacement required.",
        "06:16:00 | DISPATCHER-07| Alert acknowledged. Departure approved. Note: 'Deadline critical.'",
        "06:16:10 | DRIVER-33    | Departed depot. Trip #T-3301 started.",
        # Day 1 — incident
        "09:42:17 | DRIVER-33    | Emergency braking attempted on Highway-9. Brakes unresponsive.",
        "09:42:19 | SYSTEM       | Vehicle #V-31 collision detected. Impact speed: 38 mph.",
        "09:43:00 | EMERGENCY    | Incident reported. Driver minor injuries. Cargo lost.",
        # Post-incident
        "10:30:00 | INVESTIGATOR | Brake pads found at 6% friction material remaining. "
        "Failure consistent with pre-existing wear.",
        "10:35:00 | INVESTIGATOR | Inspection form timestamps inconsistent with physical checks.",
        "10:40:00 | INVESTIGATOR | MECHANIC-05 location data contradicts countersignature.",
    ],
    "ground_truth": {
        "anomaly_detected": True,
        "agent_id": "DRIVER-33, MECHANIC-05, DISPATCHER-07",
        "severity": "critical",
        "issue_keywords": [
            "skipped inspection",
            "fraudulent countersignature",
            "ignored brake alert",
            "cascade negligence",
            "location mismatch",
            "deadline pressure override",
        ],
    },
}
