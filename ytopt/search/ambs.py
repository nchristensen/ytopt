"""Asynchronous Model-Based Search.

Arguments of AMBS :
* ``learner``

    * ``RF`` : Random Forest (default)
    * ``ET`` : Extra Trees
    * ``GBRT`` : Gradient Boosting Regression Trees
    * ``DUMMY`` :
    * ``GP`` : Gaussian process

* ``liar-strategy``

    * ``cl_max`` : (default)
    * ``cl_min`` :
    * ``cl_mean`` :

* ``acq-func`` : Acquisition function

    * ``LCB`` :
    * ``EI`` :
    * ``PI`` :
    * ``gp_hedge`` : (default)
"""


import signal

from ytopt.search.optimizer import Optimizer
from ytopt.search import Search
from ytopt.search import util

logger = util.conf_logger('ytopt.search.hps.ambs')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 1    # How many jobs to complete between optimizer checkpoints
EXIT_FLAG = False


def on_exit(signum, stack):
    global EXIT_FLAG
    EXIT_FLAG = True


class AMBS(Search):
    def __init__(self, learner='RF', liar_strategy='cl_max', acq_func='gp_hedge',
                 set_KAPPA=1.96, set_SEED=12345, set_NI=10, initial_observations=None, error_flag_val=None, **kwargs):
        super().__init__(**kwargs)

        logger.info("Initializing AMBS")

        self.error_flag_val = error_flag_val
        if initial_observations is not None:
            initial_observations = self.forbid_invalid(initial_observations)



        self.optimizer = Optimizer(
            num_workers=self.num_workers,
            space=self.problem.input_space,
            learner=learner,
            acq_func=acq_func,
            liar_strategy=liar_strategy,
            set_KAPPA=set_KAPPA,
            set_SEED=set_SEED,
            set_NI=set_NI,
            initial_observations=initial_observations
        )

    @staticmethod
    def _extend_parser(parser):
        parser.add_argument('--learner',
                            default='RF',
                            choices=["RF", "ET", "GBRT", "DUMMY", "GP"],
                            help='type of learner (surrogate model)'
                            )
        parser.add_argument('--liar-strategy',
                            default="cl_max",
                            choices=["cl_min", "cl_mean", "cl_max"],
                            help='Constant liar strategy'
                            )
        parser.add_argument('--acq-func',
                            default="gp_hedge",
                            choices=["LCB", "EI", "PI", "gp_hedge"],
                            help='Acquisition function type'
                            )
        parser.add_argument('--set-KAPPA',
                            default=1.96,
                            type=float,
                            help='Acquisition function kappa'
                            )
        parser.add_argument('--set-SEED',
                            default=12345,
                            type=int,
                            help='Seed random_state'
                            )
        parser.add_argument('--set-NI',
                            default=10,
                            type=int,
                            help='Set n inital points'
                            )
        return parser


    def forbid_invalid(self, results):
        from ConfigSpace import ConfigurationSpace, ForbiddenEqualsClause, ForbiddenAndConjunction
        if self.error_flag_val is not None and isinstance(self.problem.input_space, ConfigurationSpace):
            valid_results = []
            for result in results:
                if result[1] == self.error_flag_val:
                    # Forbid the point
                    logger.info("Forbidding a point:", result[0])

                    try:
                        forbidden_clauses = [ForbiddenEqualsClause(self.problem.input_space[key], val) for key, val in result[0].items()]
                        forbidden_conjunction = ForbiddenAndConjunction(*forbidden_clauses)
                        self.problem.input_space.add_forbidden_clause(forbidden_conjunction)
                    except ValueError as e:
                        print(f"Forbidding point {result} was attemped, but failed due to a ValueError")
                        print(e)
                        
                    if hasattr(self, "optimizer"):
                        print(self.optimizer.space)
                        assert self.problem.input_space == self.optimizer.space
                else:                   
                    valid_results.append(result)
            results = valid_results

        return results


    def main(self):

        logger.info(f"Generating {self.num_workers} initial points...")
        XX = self.optimizer.ask_initial(n_points=self.num_workers)

        if not hasattr(self.evaluator,
                       "executor") or self.evaluator.executor is not None:
            timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
            chkpoint_counter = 0
            num_evals = 0

            self.evaluator.add_eval_batch(XX)
            # MAIN LOOP
            for elapsed_str in timer:
                logger.info(f"Elapsed time: {elapsed_str}")
                results = list(self.evaluator.get_finished_evals())
                num_evals += len(results)
                chkpoint_counter += len(results)

                results = self.forbid_invalid(results)

                if EXIT_FLAG or num_evals >= self.max_evals:
                    break
                if results:
                    logger.info(
                        f"Refitting model with batch of {len(results)} evals")
                    self.optimizer.tell(results)
                    logger.info(
                        f"Drawing {len(results)} points with strategy {self.optimizer.liar_strategy}")
                    for batch in self.optimizer.ask(n_points=len(results)):
                        self.evaluator.add_eval_batch(batch)
                if chkpoint_counter >= CHECKPOINT_INTERVAL:
                    self.evaluator.dump_evals()
                    chkpoint_counter = 0

            logger.info('Hyperopt driver finishing')
            self.evaluator.dump_evals()
            self.evaluator.shutdown()

from autotune import TuningProblem # Is an entire new class necessary?
class LibEnsembleTuningProblem(TuningProblem):

    def __init__(self, *args, **kwargs):
        if "wrapper_script" in kwargs:
            self.wrapper_script = kwargs["wrapper_script"]
            del kwargs["wrapper_script"]
        else:
            self.wrapper_script = None

        super().__init__(*args, **kwargs)    


    def libE_objective(self, H, persis_info, sim_specs, libE_info):
        import numpy as np
        import time
        from libensemble.executors import Executor, MPIExecutor

        executor = libE_info["executor"]
 
        if getattr(self, "start_time", None) is None:
            self.start_time = time.time()

        point = {}
        for field in sim_specs['in']:
            point[field] = np.squeeze(H[field])

        if self.wrapper_script is None:
            y = self.objective(point)#, sim_specs['in'], libE_info['workerID'])  # ytopt objective wants a dict
        elif isinstance(executor, MPIExecutor):
            # Do something similar to below, but communicate via disk instead.
            from pickle import dump, load
            import secrets

            filename = secrets.token_hex(nbytes=4) + ".pkl"

            worker_id = 0
            with open(filename, "xb") as file:
                dump([self.objective, point, worker_id], file)
                #file.flush()

             # Run the script using the executor
            task = executor.submit(app_name="python",
                                   #calc_type="sim",
                                   app_args= self.wrapper_script + " " + filename,
                                   stdout="out.txt",
                                   stderr="err.txt",
                                   auto_assign_gpus=True,
                                   match_procs_to_gpus=True
            )

            executor.polling_loop(task, timeout=self.objective.timeout)

            #from libensemble.tools.test_support import check_gpu_setting
            #check_gpu_setting(task, assert_setting=False, print_setting=True)

            #task.wait()  
            #logger.info("TASK STATUS:", task.success)

            #logger.info("STDOUT:", task.read_stdout())

            with open(filename, "rb") as file:
                #file.seek(0)
                y = load(file)

            if y == "ERROR" or not task.success:
                # This should probably be a property of the TuningProblem object rather
                # than the objective function.

                logger.info("TASK STATUS:", task.success)
                logger.info("STDERR:", task.read_stderr())
                y = self.objective.error_return_time
            
            #logger.info("y DETAILS:", type(y), y)

            # Cleanup the file
            import os
            os.remove(filename)

        else:

            # Allows for running the objective function in a separate python process to
            # protect the main process from crashes. Currently, all data is between the
            # called script and the calling script through shared memory. The called script must accept the
            # shared memory name as its single argument.

            from pickle import dumps, loads
            from multiprocessing import shared_memory


            # MPI within MPI is not necessarily well supported.
            if False:#not isinstance(executor, MPIExecutor):
                # Need to send rank/worker id to executor
                # Should be more portable. Not necessarily executing with MPI
                # I imagine libensemble has some kind of worker id
                # ...or maybe this is what the resource sets are for.
                import mpi4py
                mpi4py.rc.initialize = False
                import mpi4py.MPI as MPI

                if not MPI.Is_initialized():
                    MPI.Init()
                comm = MPI.COMM_WORLD
                worker_id = comm.Get_rank()
            else:
                from libensemble.resources.resources import Resources
                worker_id = Resources.resources.worker_resources.workerID
                assert worker_id is not None

            pickled_data = dumps([self.objective, point, worker_id])
            sh_mem = shared_memory.SharedMemory(create=True, size=len(pickled_data))
            sh_mem.buf[:] = pickled_data[:]

            # Run the script using the executor
            task = executor.submit(app_name="python",
                                   #calc_type="sim",
                                   app_args= self.wrapper_script + " " + sh_mem.name,
                                   stdout="out.txt",
                                   stderr="err.txt",
            )

            executor.polling_loop(task, timeout=self.objective.timeout)
            #task.wait()  

            logger.info("TASK STATUS:", task.success)
            if not task.success:
                y = self.objective.error_return_time
            #logger.info("STDERR:", task.read_stderr())
            #logger.info("STDOUT:", task.read_stdout())

            # Read the data from shared memory
            y = loads(sh_mem.buf) 
            logger.info("y DETAILS:", type(y), y)

            sh_mem.unlink()
            sh_mem.close()

        H_o = np.zeros(2, dtype=sim_specs['out'])
        H_o['RUNTIME'] = y
        H_o['elapsed_sec'] = time.time() - self.start_time

        return H_o, persis_info
        

class LibEnsembleAMBS(AMBS):

    # Call super and add more stuff for libEnsemble here
    # Note, the "problem" in kwargs must (currently) be a LibEnsembleTuningProblem object.
    def __init__(self, learner='RF', liar_strategy='cl_max', acq_func='gp_hedge',
                 set_KAPPA=1.96, set_SEED=12345, set_NI=10, initial_observations=None, 
                 libE_specs={}, evaluator="threadpool", **kwargs):

        logger.info("STARTING LibEnsembleAMBS constructor")

        #print(kwargs, flush=True)

        # For some reason this deadlocks if an mpi_comm_executor is used here. It
        # shouldn't really matter what the evaluator is set to, since it isn't used anyway.
        # Setting this to be "threadpool" by default. 
        assert evaluator != "mpi_comm_executor"
        super().__init__(learner=learner, liar_strategy=liar_strategy, acq_func=acq_func,
            set_KAPPA=set_KAPPA, set_SEED=set_SEED, set_NI=set_NI, initial_observations=initial_observations, evaluator=evaluator, **kwargs)

        if "comms" in libE_specs and libE_specs["comms"] == "local":
            # Need to specify tne number of workers if local comms are used
            assert "nworkers" in libE_specs

        if "nworkers" not in libE_specs:
            # Currently assume we're using MPI if comms is not specified to be local
            # Should probably check the command line arguments or something. Maybe
            # can detect from libE_specs
            import mpi4py
            mpi4py.rc.initialize = False
            import mpi4py.MPI as MPI
            if not MPI.Is_initialized():
                MPI.Init()
            comm = MPI.COMM_WORLD

            libE_specs["nworkers"] = comm.Get_size() - 1

        #self.path_app_name_pairs = path_app_name_pairs
        self.libE_specs = libE_specs
        #assert "nworkers" in libE_specs # For mpi, this can be set from the communicator
        #self.num_sim_workers = libE_specs["nworkers"] - 1
        #self.is_manager = is_manager
        self.output_file_base = kwargs["output_file_base"] if "output_file_base" in kwargs else "./results"


    @property
    def num_sim_workers(self):
        return self.libE_specs["nworkers"] - 1


    @staticmethod
    def parse_args():
        from libensemble.tools import parse_args

        # Parse comms, default options from commandline
        nworkers, is_manager, libE_specs, user_args_in = parse_args()
        libE_dict = {"nworkers": nworkers, "is_manager": is_manager, "libE_specs": libE_specs}


        assert len(user_args_in), "learner, etc. not specified, e.g. --learner RF"
        user_args = {}
        for entry in user_args_in:
            if entry.startswith('--'):
                if '=' not in entry:
                    key = entry.strip('--')
                    value = user_args_in[user_args_in.index(entry)+1]
                else:
                    split = entry.split('=')
                    key = split[0].strip('--')
                    value = split[1]

            user_args[key] = value

        req_settings = ['learner','max-evals']
        assert all([opt in user_args for opt in req_settings]), \
            "Required settings missing. Specify each setting in " + str(req_settings)

        return libE_dict | user_args

    def main(self):

        """
        Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
        and the ytopt findRunTime interface in a simulator function.

        Execute locally via one of the following commands (e.g. 3 workers):
           mpiexec -np 4 python run_ytopt_xsbench.py
           python run_ytopt_xsbench.py --nworkers 3 --comms local

        The number of concurrent evaluations of the objective function will be 4-1=3.
        """

        logger.info("STARTING MAIN FUNCTION")


        import os
        import glob
        import secrets
        import numpy as np
        import time

        # Import libEnsemble items for this test
        #from libensemble.libE import libE
        from libensemble import Ensemble
        from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
        from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
        from libensemble.executors import Executor, MPIExecutor

        #from ytopt_obj import init_obj  # Simulator function, calls Plopper
        #from ytopt_asktell import persistent_ytopt  # Generator function, communicates with ytopt optimizer

        #import ConfigSpace as CS
        #import ConfigSpace.hyperparameters as CSH
        #from ytopt.search.optimizer import Optimizer


        print("CREATING ENSEMBLE DIRECTORIES")
        # Set options so workers operate in unique directories
        # These should be passed in actually
        here = os.getcwd() + '/'
        if 'use_worker_dirs' not in self.libE_specs:
            self.libE_specs['use_worker_dirs'] = True
        if 'sim_dirs_make' not in self.libE_specs:
            self.libE_specs['sim_dirs_make'] = False  # Otherwise directories separated by each sim call
        if 'ensemble_dir_path' not in self.libE_specs:
            self.libE_specs['ensemble_dir_path'] = './ensembles/ensemble_' + secrets.token_hex(nbytes=4)

        # Copy or symlink needed files into unique directories
        #libE_specs['sim_dir_copy_files'] = [here + f for f in ['mmp.c', 'Materials.c', 'XSutils.c', 'XSbench_header.h']]
        #self.libE_specs['sim_dir_symlink_files'] = [here + f for f in ['mmp.c', 'Materials.c', 'XSutils.c', 'XSbench_header.h', 'exe.pl', 'plopper.py', 'processexe.pl']]

        # Declare the sim_f to be optimized, and the input/outputs
        print("CREATING SIM SPECS")
        sim_specs = {'sim_f': self.problem.libE_objective,#init_obj,
                     'in': list(self.problem.input_space.keys()),#['p0', 'p1', 'p2', 'p3', 'p4'],
                     'out': [('RUNTIME', float),('elapsed_sec', float)],
                    }

        """
        cs = CS.ConfigurationSpace(seed=1234)
        # Initialize the ytopt ask/tell interface (to be used by the gen_f)
        p0 = CSH.OrdinalHyperparameter(name='p0', sequence=[2,4, 8,16,32,64,128], default_value=64)
        # block size for openmp dynamic schedule
        p1 = CSH.OrdinalHyperparameter(name='p1', sequence=[10, 20, 40, 64, 80, 100, 128, 160, 200], default_value=100)
        # omp parallel
        p2 = CSH.CategoricalHyperparameter(name='p2', choices=["#pragma omp parallel for", " "], default_value=' ')
        # omp placement
        p3 = CSH.CategoricalHyperparameter(name='p3', choices=['cores', 'threads', 'sockets'], default_value='cores')
        # OMP_PROC_BIND
        p4= CSH.CategoricalHyperparameter(name='p4', choices=['close','spread','master'], default_value='close')

        cs.add_hyperparameters([p0, p1, p2, p3, p4])

        ytoptimizer = Optimizer(
            num_workers=num_sim_workers,
            space=cs,
            learner=user_args['learner'],
            liar_strategy='cl_max',
            acq_func='gp_hedge',
            set_KAPPA=1.96,
            set_SEED=2345,
            set_NI=10,
        )
        """

        # Declare the gen_f that will generate points for the sim_f, and the various input/outputs
        from itertools import product
        print("CREATING GEN SPECS") 
        gen_specs = {
            'gen_f': self,#persistent_ytopt, # self is a callable
            # Assume that each entry in the sampled from the ConfigSpace is a single integer.
            # This probably needs to be generalized
            'out': list(product(self.problem.input_space.keys(), [int], [(1,)])),
            #'out': [('p0', int, (1,)), ('p1', int, (1,)), ('p2', "<U24", (1,)),
            #        ('p3', "<U7", (1,)), ('p4', "<U8", (1,)), ],
            'persis_in': sim_specs['in'] + ['RUNTIME'] + ['elapsed_sec'],
            'user': {
                'ytoptimizer': self.optimizer,  # provide optimizer to generator function
                'num_sim_workers': self.num_sim_workers,
            },
        }

        alloc_specs = {
            'alloc_f': alloc_f,
            'user': {'async_return': True},
        }

        # Specify when to exit. More options: https://libensemble.readthedocs.io/en/main/data_structures/exit_criteria.html
        exit_criteria = {'gen_max': int(self.max_evals)}

        # Added as a workaround to issue that's been resolved on develop
        persis_info = add_unique_random_streams({}, self.libE_specs["nworkers"] + 1)

        # Register apps with executor. Doesn't need to be passed in as this creates a class variable which libE looks for.
        if "comms" in self.libE_specs and self.libE_specs["comms"] == "local":
            executor = MPIExecutor()
        else:
            executor = Executor()
        #for path, app_name in self.path_app_name_pairs:

        import sys
        executor.register_app(full_path=sys.executable, app_name="python")#, calc_type="sim")

        # Perform the libE run
        print("STARTING libE", self.libE_specs)

        ensemble = Ensemble(sim_specs=sim_specs, gen_specs=gen_specs, exit_criteria=exit_criteria,
                            persis_info=persis_info, alloc_specs=alloc_specs, libE_specs=self.libE_specs,
                            executor=executor)
        H, persis_info, flag = ensemble.run()

        #H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
        #                            alloc_specs=alloc_specs, libE_specs=self.libE_specs)


        # Save History array to file
        #if self.is_manager:
        #if ensemble.is_manager:
        #    print("\nlibEnsemble has completed evaluations.")
            #save_libE_output(H, persis_info, __file__, nworkers)

            #print("\nSaving just sim_specs[['in','out']] to a CSV")
            #H = np.load(glob.glob('*.npy')[0])
            #H = H[H["sim_ended"]]
            #H = H[H["returned"]]
            #dtypes = H[gen_specs['persis_in']].dtype
            #b = np.vstack(map(list, H[gen_specs['persis_in']]))
            #print(b)
            #np.savetxt('results.csv',b, header=','.join(dtypes.names), delimiter=',',fmt=','.join(['%s']*b.shape[1]))


    # Generates the points. This is essentially equivalent to the AMBS main
    # In normal ytopt, the main loop handles the point generation.
    # In persistent libensemble, the generator runs on a different process
    # ytopt is missing an abstraction for this situation.
    def __call__(self, H, persis_info, gen_specs, libE_info):
        """
        This module wraps around the ytopt generator.
        """
        import numpy as np
        from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG, EVAL_GEN_TAG
        from libensemble.tools.persistent_support import PersistentSupport


        #__all__ = ['persistent_ytopt']

        ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
        user_specs = gen_specs['user']
        ytoptimizer = user_specs['ytoptimizer']

        tag = None
        calc_in = None
        first_call = True
        first_write = True
        from os.path import exists
        if exists("../../../" + self.output_file_base + '.csv'):
            first_write = False

        fields = [i[0] for i in gen_specs['out']]

        # Send batches until manager sends stop tag
        while tag not in [STOP_TAG, PERSIS_STOP]:

            if first_call:
                ytopt_points = ytoptimizer.ask_initial(n_points=user_specs['num_sim_workers'])  # Returns a list
                batch_size = len(ytopt_points)
                first_call = False
            else:
                batch_size = len(calc_in)
                results = []
                for entry in calc_in:
                    field_params = {}
                    for field in fields:
                        field_params[field] = entry[field][0]
                    results += [(field_params, entry['RUNTIME'])]
                print('results: ', results)
                results = self.forbid_invalid(results)
                ytoptimizer.tell(results)

                ytopt_points = ytoptimizer.ask(n_points=batch_size)  # Returns a generator that we convert to a list
                ytopt_points = list(ytopt_points)[0]

            # The hand-off of information from ytopt to libE is below. This hand-off may be brittle.
            H_o = np.zeros(batch_size, dtype=gen_specs['out'])
            for i, entry in enumerate(ytopt_points):
                for key, value in entry.items():
                    H_o[i][key] = value

            # This returns the requested points to the libE manager, which will
            # perform the sim_f evaluations and then give back the values.
            tag, Work, calc_in = ps.send_recv(H_o)
            print('received:', calc_in, flush=True)

            if calc_in is not None:
                if len(calc_in):
                    b = []
                    for entry in calc_in[0]:
                        try: 
                            b += [str(entry[0])]
                        except: 
                            b += [str(entry)]

 
                    with open("../../../" + self.output_file_base + '.csv', 'a') as f:
                    #with open('../../results.csv', 'a') as f:
                        if first_write:
                            f.write(",".join(calc_in.dtype.names)+ "\n")
                            f.write(",".join(b)+ "\n")
                            first_write = False
                        else:
                            f.write(",".join(b)+ "\n")

        return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG
        
    

if __name__ == "__main__":
    args = AMBS.parse_args()
    search = AMBS(**vars(args))
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)
    search.main()
