class ResultsLogger:


    def  __init__(self, config: str='', project_name: str='testing',
                   test_mode: bool = False, exp_type: str = '',
                    fold: str = ''):
        self.config = config
        self.test_mode = test_mode
        self.exp_type = exp_type
        self.fold = fold
        if not self.test_mode:
            import wandb
            # with open(config, 'rt') as fp:
            #     config = yaml.safe_load(fp)
            self.config = config
            self.run = wandb.init(project=project_name, config=config)
            self.run.name = f"{self.run.name}_{self.exp_type}_{self.fold}"

    def log(self, 
            epoch,
            epoch_loss: float = 0.0,
            epoch_mae: float = 0.0, 
            epoch_mse: float = 0.0,
            test_loss: float =0.0,
            test_mae: float = 0.0, 
            test_error: float =0.0,
            epoch_time: float = 0.0,
            exp_type: str = ''):
        if not self.test_mode:
            self.run.log({
            'epoch': epoch,
            'Train loss': round(epoch_loss,4),
            'Train mse': round(epoch_mse,4),
            'Train mae': round(epoch_mae,4),
            'Test loss': round(test_loss,4),
            'Test mse': round(test_error,4),
            'Test mae': round(test_mae,4),
            'Time per epoch': round(epoch_time,4),
            'Experiment type': exp_type
             })
            

            if epoch % 10 == 0:
                print(f'epoch: {epoch}, train_loss: {epoch_loss:.4f}, train_mse: {epoch_mse:.4f}, train_mae: {epoch_mae:.4f}, test_loss: {test_loss:.4f},test_mse: {test_error:.4f}, test_mae: {test_mae:.4f}, time per epoch: {epoch_time:.2f}')
                

        else:
            print(f'epoch: {epoch}, train_loss: {epoch_loss:.4f}, train_mse: {epoch_mse:.4f}, train_mae: {epoch_mae:.4f}, test_loss: {test_loss:.4f},test_mse: {test_error:.4f}, test_mae: {test_mae:.4f}, time per epoch: {epoch_time:.2f}')

    def retrieve_run_name(self):
        return self.run.name