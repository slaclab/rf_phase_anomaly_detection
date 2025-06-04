import time
from multiprocessing import Manager, Process
from mp_logging import run_logger_process, create_worker_logger
from process import ProcessManager
from k2eg_process import K2EGProcess, read_pv_list_from_file
from process_b import ProcessB
from process_c import ProcessC


if __name__ == '__main__':
    list_of_pvs = read_pv_list_from_file('resources/very_short_pv_list.txt')

    # Create an instance of the Manager
    with Manager() as manager:
        queue_one = manager.Queue()
        queue_two = manager.Queue()
        queue_log = manager.Queue()  # for logging only

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

        process_objects = [
            K2EGProcess(
                queue_one,
                pv_list=list_of_pvs,
                snapshot_period_ms=1000,
                logging_kwargs=logging_kwargs.copy()
            ),
            ProcessB(queue_one, queue_two, logging_kwargs=logging_kwargs.copy()),
            ProcessC(queue_two, logging_kwargs=logging_kwargs.copy()),
        ]

        with ProcessManager(process_objects=process_objects) as pm:
            while pm.is_running:
                time.sleep(1)
                main_logger.debug('pm loop')

        logger_process.join()

    print('Processes are done.')
