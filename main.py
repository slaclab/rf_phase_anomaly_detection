import time
import argparse
from multiprocessing import Manager, Process
from mp_logging import run_logger_process, create_worker_logger
from process import ProcessManager
from k2eg_process import K2EGProcess, read_pv_list_from_file
from k2eg_spoof_process import K2EGSpoofProcess
from process_b import ProcessB
from process_c import ProcessC


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run with real or spoofed k2eg data.")
    parser.add_argument(
        '-skd', '--spoof_k2eg_data',
        action='store_true',
        help='Use spoofed k2eg data (random values) instead of real pv data.'
    )       
    args = parser.parse_args()

    list_of_pvs = read_pv_list_from_file('resources/very_short_pv_list.txt')

    # Create an instance of the Manager
    with Manager() as manager:
        # create queues for inter-process communication
        queue_one = manager.Queue()
        queue_two = manager.Queue()
        queue_log = manager.Queue()  # for logging only

        # logging configuration
        logging_kwargs = default_logging_kwargs = {
            'queue': queue_log,
            'logger_name': 'test_log',
            'log_level': 10,  # 10 is DEBUG
            'log_stdout': True
        }
        logger_process = Process(target=run_logger_process, kwargs=logging_kwargs)
        logger_process.start()
        main_logger = create_worker_logger(
            queue=logging_kwargs['queue'],
            logger_name='main',
            log_level=logging_kwargs['log_level'],
            log_stdout=logging_kwargs['log_stdout']
        )

        # create classes to be turned into processes
        # it helps to put them in order
        if args.spoof_k2eg_data:
            k2eg_proc = K2EGSpoofProcess(
                queue=queue_one,
                pv_list=list_of_pvs,
                n_emits=10,
                emit_rate_hz=1,
                logging_kwargs=logging_kwargs.copy()
            )
        else:
            k2eg_proc = K2EGProcess(
                queue=queue_one,
                pv_list=list_of_pvs,
                snapshot_period_ms=1000,
                logging_kwargs=logging_kwargs.copy()
            )
    
        process_objects = [
            k2eg_proc,
            ProcessB(queue_one, queue_two, logging_kwargs=logging_kwargs.copy()),
            ProcessC(queue_two, logging_kwargs=logging_kwargs.copy()),
        ]

        # use ProcessManager to handle start and join of processes
        with ProcessManager(process_objects=process_objects) as pm:
            while pm.is_running:
                time.sleep(1)
                main_logger.debug('pm loop')

        logger_process.join()  # wait for the logger to finish last

    print('Processes are done.')
