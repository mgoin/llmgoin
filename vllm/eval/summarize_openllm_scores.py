import json
import glob

tasks = [
    "arc_challenge",
    "winogrande",
    "truthfulqa_mc2",
    "hellaswag",
    "mmlu",
    "gsm8k"
]

def extract_scores(task_name, json_data):
    scores = {}
    if 'results' in json_data and task_name in json_data['results']:
        scores.update(json_data['results'][task_name])
    if 'n-shot' in json_data and task_name in json_data['n-shot']:
        scores['num_fewshot'] = json_data['n-shot'][task_name]
    return scores

def scrape_scores(task_list):
    scores = {}
    for task in task_list:
        pattern = f"results/{task}/**/*.json"
        files = glob.glob(pattern, recursive=True)
        if not files:
            print(f"No files found for task: {task}")
            continue
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                task_scores = extract_scores(task, data)
                if task_scores:
                    scores[file] = task_scores
                else:
                    print(f"No scores found in file: {file}")
    return scores

def main():
    scores = scrape_scores(tasks)
    for file, score in scores.items():
        print(f"Scores from {file}:")
        for metric, value in score.items():
            print(f"  {metric}: {value}")
        print()

if __name__ == "__main__":
    main()
