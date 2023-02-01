            sn_utils.set_seed(0)
            # TODOBRW my version
            #self.argparseArgs = parse_app_args(argv=sys.argv, common_parser_fn=add_args, run_parser_fn=add_run_args)

            # TODOBRW My update.
            # Arg Handler -- note: no validity checking done here
            argv = sys.argv[1:]
            self.argparseArgs = parse_app_args(argv=argv, common_parser_fn=add_args)
            print(f'self.argparseArgs:\n{self.argparseArgs}')


            parsed = Path(config_file)
            heinitialize(config_path=str(parsed.parent))
            overrides = ["mode=train", "run.id=run_id", "run.distributed=False", "data.data_directory=/nvmedata/ANL/cosmictagger/", "data.downsample=0", "framework=torch", "run.compute_mode=RDU", "run.minibatch_size=1", "run.iterations=1", "run.precision=3"]
            config = compose(parsed.name, overrides=overrides)

            self.hydraArgs = config
