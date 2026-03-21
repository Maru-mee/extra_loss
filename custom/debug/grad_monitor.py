import pandas as pd
import os
import json

import webbrowser

def generate_html(input_csv="grad_log.csv", output_html="grad_monitor.html"):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    steps = sorted(df['step'].unique().tolist())
    names = sorted(df['name'].unique().tolist())
    
    colors = [
        "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4",
        "#42D4F4", "#F032E6", "#BFEF45", "#FABEBE", "#469990", "#E6BEFF"
    ]
    
    datasets = []
    for i, name in enumerate(names):        
        subset = df[df['name'] == name].copy()
        subset['mean'] = subset['mean'].rolling(window=10, min_periods=1).mean()
        subset = subset.iloc[::5]
        
        subset_dict = dict(zip(subset['step'], subset['mean']))
        data = [subset_dict.get(s, None) for s in steps]
        
        color = colors[i % len(colors)]
        is_dash = [5, 5] if "batch" in name or "pair" in name else []

        datasets.append({
            "label": name,
            "data": data,
            "borderColor": color,
            "backgroundColor": color,
            "borderDash": is_dash,
            "borderWidth": 2,
            "pointRadius": 0,
            "fill": False,
            "tension": 0.1,
            "spanGaps": True
        })

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>body {{ background: #1a1a1a; color: #eee; font-family: sans-serif; }}</style>
</head>
<body>
    <div style="width: 95%; margin: auto;">
        <canvas id="c"></canvas>
    </div>
    <script>
        const ctx = document.getElementById('c');
        new Chart(ctx, {{
            type: 'line',
            data: {{ 
                labels: {json.dumps(steps)}, 
                datasets: {json.dumps(datasets)} 
            }},
            options: {{
                animation: false,
                events: [],
                scales: {{
                    y: {{ 
                        type: 'logarithmic',
                        grid: {{ color: '#444' }},
                        ticks: {{ color: '#ccc' }}
                    }},
                    x: {{ 
                        grid: {{ color: '#444' }},
                        ticks: {{ color: '#ccc' }}
                    }}
                }},
                plugins: {{
                    legend: {{ 
                        position: 'top',
                        labels: {{ color: '#ccc', font: {{ size: 12 }} }} 
                    }},
                    tooltip: {{ 
                        enabled: false
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated: {output_html}")
    
    # 絶対パスを取得してブラウザで開く
    webbrowser.open("file://" + os.path.realpath(output_html))

if __name__ == "__main__":
    generate_html()