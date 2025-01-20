import optuna
import torch
import wandb
from config import TrainingConfig, HPSearchSpace
from training_utils import train_epoch, evaluate_model, LossComputer, EarlyStopping
from typing import Dict, Any

class HPOptimizer:
    def __init__(
        self,
        base_config: TrainingConfig,
        search_space: HPSearchSpace,
        n_trials: int = 50,
        study_name: str = "semi_supervised_learning"
    ):
        self.base_config = base_config
        self.search_space = search_space
        self.n_trials = n_trials
        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name
        )
        
    def objective(self, trial: optuna.Trial) -> float:
        # Create trial-specific config
        config = TrainingConfig(
            cls_weight=trial.suggest_float(
                "cls_weight",
                *self.search_space.cls_weight_range
            ),
            consistency_weight=trial.suggest_float(
                "consistency_weight",
                *self.search_space.consistency_weight_range
            ),
            confidence_alpha=trial.suggest_float(
                "confidence_alpha",
                *self.search_space.confidence_alpha_range
            ),
            base_lr=trial.suggest_float(
                "learning_rate",
                *self.search_space.lr_range,
                log=True
            )
        )
        
        # Initialize wandb for this trial
        run = wandb.init(
            project="semi_supervised_learning",
            config=vars(config),
            reinit=True
        )
        
        try:
            # Initialize models, dataloaders, etc.
            student_net = self.initialize_model(config)
            teacher_net = self.initialize_model(config)
            train_loader, val_loader = self.get_dataloaders(config)
            
            # Training setup
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer = torch.optim.Adam(student_net.parameters(), lr=config.base_lr)
            loss_computer = LossComputer(config)
            early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta
            )
            
            best_accuracy = 0.0
            
            for epoch in range(config.epochs):
                # Train epoch
                train_metrics = train_epoch(
                    student_net=student_net,
                    teacher_net=teacher_net,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    loss_computer=loss_computer,
                    device=device
                )
                
                # Evaluate
                accuracy, val_loss = evaluate_model(
                    model=student_net,
                    val_loader=val_loader,
                    device=device
                )
                
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "accuracy": accuracy,
                    "val_loss": val_loss,
                    **train_metrics
                })
                
                # Update best accuracy
                best_accuracy = max(best_accuracy, accuracy)
                
                # Early stopping
                if early_stopping(val_loss):
                    break
                    
            run.finish()
            return best_accuracy
            
        except Exception as e:
            run.finish()
            raise e
            
    def run_optimization(self) -> Dict[str, Any]:
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        # Train final model with best parameters
        final_config = TrainingConfig(**best_params)
        final_accuracy = self.train_final_model(final_config)
        
        return {
            "best_params": best_params,
            "best_trial_accuracy": best_value,
            "final_model_accuracy": final_accuracy
        }
        
    def train_final_model(self, config: TrainingConfig) -> float:
        # Similar to objective(), but with final training and model saving
        pass
        
    @staticmethod
    def initialize_model(config: TrainingConfig):
        # Initialize model based on config
        pass
        
    @staticmethod
    def get_dataloaders(config: TrainingConfig):
        # Initialize dataloaders based on config
        pass
