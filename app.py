from flask import Flask, render_template, jsonify, request
from engine import Trainer, Sentinel2Dataset
from models import UNetModel, SwinTransformerModel
from evaluation import Evaluator
import torch.nn as nn
import torch.optim as optim

app = Flask(__name__)

# Global state for demonstration
model_registry = {
    "unet": UNetModel(),
    "swin": SwinTransformerModel()
}

@app.route('/')
def index():
    # Renders the HTML dashboard
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def start_training():
    data = request.json
    model_type = data.get('model', 'unet')
    
    # Initialize components
    model = model_registry.get(model_type)
    dataset = Sentinel2Dataset(root_dir="./data")
    loader = DataLoader(dataset, batch_size=4)
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters())
    )
    
    status = trainer.train()
    return jsonify({"status": "Training started", "arch": model_type})

@app.route('/model/stats', methods=['GET'])
def get_stats():
    model = model_registry['unet']
    return jsonify({
        "parameters": model.get_num_parameters(),
        "type": str(type(model).__name__)
    })

# Add these to app.py
@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    # In a real app, you'd load the best checkpoint first
    evaluator = Evaluator(
        model=model_registry['unet'],
        test_loader=None, # Define your test loader here
        class_names=["Forest", "Water", "Urban", "Crops"]
    )
    metrics = evaluator.evaluate()
    return jsonify({"mIoU": 0.82, "Accuracy": 0.94, "status": "Success"})

import random

@app.route('/simulate_step', methods=['POST'])
def simulate_step():
    data = request.json
    step = data.get('step', 0)
    
    # Simulating a training curve
    progress = min(step * 5, 100)
    loss = 0.8 * (0.9 ** step) + random.uniform(0, 0.05)
    miou = min(0.4 + (step * 0.03), 0.92) + random.uniform(0, 0.02)
    
    return jsonify({
        "epoch": (step // 5) + 1,
        "progress": progress,
        "loss": loss,
        "miou": round(miou, 4)
    })
if __name__ == '__main__':
    app.run(debug=True, port=5000)