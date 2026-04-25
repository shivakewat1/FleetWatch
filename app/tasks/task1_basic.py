TASK = {
    "input_logs": [
        "Vehicle #101 departed depot at 08:00",
        "Vehicle #101 speed: 72 mph on highway segment A",
        "Vehicle #102 idle for 45 minutes at waypoint B",
        "Vehicle #103 hard braking event detected at 09:15",
    ],
    "ground_truth": {
        "risk_level": "medium",
        "flagged_vehicles": ["#101", "#103"],
        "recommended_action": "review",
    },
}
