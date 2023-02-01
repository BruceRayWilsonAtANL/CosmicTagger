            parsed = Path(config_file)
            heinitialize(config_path=str(parsed.parent))
            overrides = ["mode=train", "run.id=run_id", "run.distributed=False", "data.data_directory=/nvmedata/ANL/cosmictagger/", "data.downsample=0", "framework=torch", "run.compute_mode=CPU", "run.minibatch_size=1", "run.iterations=1", "run.precision=3"]
            config = compose(parsed.name, overrides=overrides)

            self.hydraArgs = config

            self.argparseArgs = None
