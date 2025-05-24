# Create and save the CSV file
csv_data = {
    "Task Name": [
        "Framework Dev",
        "Multi-terrain Eval",
        "LLM Integration",
        "Large-scale Experiments",
        "Paper Writing"
    ],
    "Start Date": [
        "2025-05-13",
        "2025-06-11",
        "2025-07-11",
        "2025-09-01",
        "2025-10-01"
    ],
    "End Date": [
        "2025-06-10",
        "2025-07-10",
        "2025-08-31",
        "2025-09-30",
        "2025-10-31"
    ],
    "Status": [
        "Completed",
        "Completed",
        "In Progress",
        "Planned",
        "Planned"
    ]
}

df_csv = pd.DataFrame(csv_data)
csv_path = "/mnt/data/project_timeline.csv"
df_csv.to_csv(csv_path, index=False)

csv_path
