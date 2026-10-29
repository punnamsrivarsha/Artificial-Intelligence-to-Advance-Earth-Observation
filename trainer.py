class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            # ... training loop logic ...
            
            val_loss = self.validate()
            
            # Step the scheduler
            self.scheduler.step()
            
            # Check Early Stopping
            if self.early_stopping.update(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, masks in self.val_loader:
                outputs = self.model(images.to(self.device))
                loss = self.criterion(outputs, masks.to(self.device))
                total_loss += loss.item()
        return total_loss / len(self.val_loader)