TASK = {
    "task_id": "task5-collusion",
    "task_description": (
        "Three agents — DRIVER-41, DRIVER-42, and FUEL-MANAGER-02 — are colluding "
        "to siphon fuel by inflating fuel purchase records, splitting phantom mileage "
        "across vehicles, and using a shell vendor account to launder the overcharges. "
        "The scheme has been running for 6 weeks. Expose the full collusion network."
    ),
    "input_logs": [
        # Week 1 anomaly seed
        "2024-02-05 | DRIVER-41      | Fuel purchase: 87 gallons @ Station-GX. Trip mileage: 310 mi.",
        "2024-02-05 | SYSTEM         | Vehicle #V-41 tank capacity: 60 gallons. Purchase exceeds capacity.",
        "2024-02-05 | FUEL-MANAGER-02| Purchase approved. Vendor: QuickFuel-GX (ID: VND-9921).",
        # Week 2
        "2024-02-12 | DRIVER-42      | Fuel purchase: 91 gallons @ Station-GX. Trip mileage: 295 mi.",
        "2024-02-12 | SYSTEM         | Vehicle #V-42 tank capacity: 60 gallons. Purchase exceeds capacity.",
        "2024-02-12 | FUEL-MANAGER-02| Purchase approved. Vendor: QuickFuel-GX (ID: VND-9921).",
        # Week 3 — mileage split
        "2024-02-19 | DRIVER-41      | Mileage log: 180 mi on Vehicle #V-41.",
        "2024-02-19 | DRIVER-42      | Mileage log: 175 mi on Vehicle #V-42.",
        "2024-02-19 | SYSTEM         | GPS telemetry: V-41 actual miles: 94. V-42 actual miles: 88.",
        "2024-02-19 | FUEL-MANAGER-02| Monthly fuel report submitted. Total variance flagged: $0.",
        # Week 4 — vendor investigation
        "2024-02-26 | FLEET-MGMT     | Vendor audit initiated for QuickFuel-GX (VND-9921).",
        "2024-02-26 | FUEL-MANAGER-02| Audit response submitted: 'Pricing reflects bulk rate agreement.'",
        "2024-02-27 | FLEET-MGMT     | VND-9921 registration address matches personal address of DRIVER-41.",
        # Week 5 — communication evidence
        "2024-03-04 | COMMS-LOG      | Internal message DRIVER-41 → DRIVER-42: "
        "'Split it 50/50 again, FM-02 will clear it.'",
        "2024-03-04 | COMMS-LOG      | Internal message DRIVER-42 → FUEL-MANAGER-02: "
        "'Same as last week, add 30 on each.'",
        # Week 6 — financial summary
        "2024-03-11 | FINANCE        | 6-week fuel overcharge estimate: $14,820.",
        "2024-03-11 | FINANCE        | Payments to VND-9921 traced to joint account held by DRIVER-41 "
        "and FUEL-MANAGER-02.",
    ],
    "ground_truth": {
        "anomaly_detected": True,
        "agent": "DRIVER-41, DRIVER-42, FUEL-MANAGER-02",
        "severity": "critical",
        "keywords": [
            "fuel siphoning",
            "inflated purchase records",
            "phantom mileage",
            "shell vendor",
            "collusion network",
            "financial fraud",
            "tank capacity exceeded",
            "GPS mileage mismatch",
        ],
    },
}
