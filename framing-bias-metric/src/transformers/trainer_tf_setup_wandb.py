def setup_wandb(self):
    """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.com/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different
                project.
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely.
        """
    logger.info(
        'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
        )
    combined_dict = {**self.model.config.to_dict(), **self.args.
        to_sanitized_dict()}
    wandb.init(project=os.getenv('WANDB_PROJECT', 'huggingface'), config=
        combined_dict, name=self.args.run_name)
