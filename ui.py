import gradio as gr
import requests
import pandas as pd
import json
import uuid

ENV_BASE_URL = "http://127.0.0.1:7860"

def new_session():
    return str(uuid.uuid4())

def load_task(task_id, sid):
    try:
        task_num = int(task_id.split()[1])
        r = requests.post(f"{ENV_BASE_URL}/api/reset", json={"task_id": task_num, "session_id": sid})
        if r.status_code != 200:
            return f"Error: {r.text}", "", "", pd.DataFrame(), f"Error loading task {task_num}"
            
        data = r.json()
        obs = data["observation"]
        return obs["task_description"], obs["schema"], obs["hint"], pd.DataFrame(), "Task loaded. Write SQL below."
    except Exception as e:
        return str(e), "", "", pd.DataFrame(), "Error loading task"

def run_sql(sql, sid):
    if not sql.strip():
        return pd.DataFrame(), "Please enter a SQL query.", "Error"
    try:
        r = requests.post(f"{ENV_BASE_URL}/api/step", json={"action": sql, "session_id": sid})
        data = r.json()
        obs = data.get("observation", {})
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        
        df = pd.DataFrame(obs.get("result_preview", []))
        breakdown = obs.get("reward_breakdown", {})
        
        feedback = f"🎯 Reward: {reward:.2f} / 1.0\n"
        if breakdown:
            feedback += f"Columns: {breakdown.get('column_score',0):.2f}, Rows: {breakdown.get('row_score',0):.2f}, Values: {breakdown.get('value_score',0):.2f}"
        if "error" in obs:
            feedback += f"\n\n⚠️ Error: {obs['error']}"
        
        return df, feedback, "✅ SOLVED!" if done else "Keep trying!"
    except Exception as e:
         return pd.DataFrame(), str(e), "Error"

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
        gr.Markdown("# 📊 SQL Analyst OpenEnv")
        gr.Markdown("An interactive environment to test SQL generation. Load a task, read the schema, and write a query answering the business question.")
        
        sid = gr.State(new_session)
        
        with gr.Group():
            gr.Markdown("### Step 1: Select a Task")
            with gr.Row():
                task_dropdown = gr.Dropdown(choices=["Task 1 (Easy)", "Task 2 (Medium)", "Task 3 (Hard)", "Task 4 (Medium)", "Task 5 (Hard)"], value="Task 1 (Easy)", label="Available Tasks", show_label=False)
                btn_load = gr.Button("🔄 Load Task", variant="primary")
        
        with gr.Row():
            desc = gr.Textbox(label="🎯 Business Question", interactive=False, lines=2)
            
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 2: Understand the Data")
                schema = gr.Code(label="Database Schema", language="sql", interactive=False)
                with gr.Accordion("💡 Need a hint?", open=False):
                    hint = gr.Textbox(show_label=False, interactive=False)
            
            with gr.Column():
                gr.Markdown("### Step 3: Write & Execute SQL")
                sql_input = gr.Code(label="SQL Editor", language="sql", lines=12)
                btn_run = gr.Button("🚀 Execute Query", variant="primary")
                
                status_out = gr.Markdown("Waiting for query...")
                feedback_out = gr.Textbox(label="Evaluation Score & Feedback", interactive=False, lines=3)
                
        with gr.Group():
            gr.Markdown("### Step 4: Review Results")
            grid_out = gr.Dataframe(label="Result Preview (First 5 Rows)", interactive=False)

        btn_load.click(load_task, inputs=[task_dropdown, sid], outputs=[desc, schema, hint, grid_out, feedback_out])
        btn_run.click(run_sql, inputs=[sql_input, sid], outputs=[grid_out, feedback_out, status_out])
        
    return demo
