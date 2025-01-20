import wandb
from config import TrainingConfig, HPSearchSpace
from hpo_script import HPOptimizer

def main():
    # Initialize base configuration
    base_config = TrainingConfig()
    
    # Define hyperparameter search space
    search_space = HPSearchSpace()
    
    # Initialize wandb
    wandb.init(project="semi_supervised_learning")
    
    # Create optimizer
    optimizer = HPOptimizer(
        base_config=base_config,
        search_space=search_space,
        n_trials=50,
        study_name="semi_supervised_learning"
    )
    
    # Run optimization
    results = optimizer.run_optimization()
    
    # Log final results
    wandb.log({
        "best_parameters": results["best_params"],
        "best_trial_accuracy": results["best_trial_accuracy"],
        "final_model_accuracy": results["final_model_accuracy"]
    })
    
    print("Best parameters:", results["best_params"])
    print("Best accuracy:", results["best_trial_accuracy"])
    print("Final model accuracy:", results["final_model_accuracy"])

if __name__ == "__main__":
    main()
