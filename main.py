import time
from multiprocessing import Manager, Process
from mp_logging import run_logger_process, create_worker_logger
from process import ProcessManager
from process_a import ProcessA
from process_b import ProcessB
from process_c import ProcessC


if __name__ == '__main__':
    # Create an instance of the Manager
    with Manager() as manager:
        # Create a queue within the context of the manager
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
            ProcessA(queue_one, n=5, logging_kwargs=logging_kwargs.copy()),
            ProcessB(queue_one, queue_two, logging_kwargs=logging_kwargs.copy()),
            ProcessC(queue_two, logging_kwargs=logging_kwargs.copy()),
        ]

        with ProcessManager(process_objects=process_objects) as pm:
            while pm.is_running:
                time.sleep(1)
                main_logger.debug('pm loop')

        logger_process.join()

    print('Processes are done.')
