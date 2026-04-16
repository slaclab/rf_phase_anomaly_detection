import time
import signal
import argparse
from multiprocessing import Manager, Process, Pipe
from mp_logging import run_logger_process, create_worker_logger
from process import ProcessManager
from process_a import K2EGProcess
from utilities import read_pv_list_from_file
from machine_interface.k2eg_spoof_process import K2EGSpoofProcess
from machine_interface.k2eg_spoof_anomaly_process import K2EGSpoofAnomalyProcess, constant_readings_dict
from process_b import ProcessB
from process_c import ProcessC
from process_i import ProcessI

def main(args: argparse.Namespace):
    # Create an instance of the Manager
    with Manager() as manager:
        # create queues for inter-process communication
        queue_one = manager.Queue()
        queue_two = manager.Queue()
        queue_log = manager.Queue()  # for logging only
        queue_ins = manager.Queue()  # for communicating with GUI
        # create a pipe for stopping process_a
        pipe_a, pipe_main = Pipe()  # receiver_pipe, sender_pipe

        main_logger_name = None if args.disable_file_logging else "kad_main"

        # logging configuration
        logging_kwargs = default_logging_kwargs = {
            "queue": queue_log,
            "logger_name": main_logger_name,
            "log_level": args.log_level,  # 10 is DEBUG
            "log_stdout": not args.disable_stdout_logging,
        }

        logger_process = Process(target=run_logger_process, kwargs=logging_kwargs)
        logger_process.start()

        main_logger = create_worker_logger(
            queue=logging_kwargs["queue"],
            logger_name=main_logger_name,
            log_level=logging_kwargs["log_level"],
            log_stdout=logging_kwargs["log_stdout"],
        )
        main_logger.info(args)

        main_logger.info(
            f"LoggerProcess running on pid {logger_process.pid} with parent pid {logger_process._parent_pid}"
        )

        instrument_po = ProcessI(queue_ins, use_k2eg=args.use_k2eg, logging_kwargs=logging_kwargs.copy())
        instrument_process = Process(target=instrument_po, args=())
        instrument_process.start()

        main_logger.info(
            f"Instr. Process running on pid {instrument_process.pid} with parent pid {instrument_process._parent_pid}"
        )

        # create classes to be turned into processes
        # it helps to put them in order
        if args.spoof_k2eg_data:
            list_of_pvs = read_pv_list_from_file("resources/pv_list.txt")
            k2eg_proc = K2EGSpoofProcess(
                queue=queue_one, pv_list=list_of_pvs, n_emits=400,
                emit_rate_hz=1, queue_inst=queue_ins, logging_kwargs=logging_kwargs.copy()
            )
        elif args.spoof_k2eg_data_anomaly:
            k2eg_proc = K2EGSpoofAnomalyProcess(
                queue=queue_one, n_emits=17, emit_anomaly_every_n_iterations=6,
                queue_inst=queue_ins, logging_kwargs=logging_kwargs.copy()
            )
            list_of_pvs = list(constant_readings_dict.keys())
        else:
            list_of_pvs = read_pv_list_from_file("resources/pv_list.txt")
            k2eg_proc = K2EGProcess(
                queue=queue_one, pv_list=list_of_pvs, snapshot_period_ms=1000,
                main_pipe=pipe_a, logging_kwargs=logging_kwargs.copy()
            )

        process_objects = [
            k2eg_proc,
            ProcessB(queue_one, queue_two,
                     pv_list=list_of_pvs, queue_inst=queue_ins, logging_kwargs=logging_kwargs.copy()),
            ProcessC(queue_two, queue_inst=queue_ins, logging_kwargs=logging_kwargs.copy()),
        ]

        # use ProcessManager to handle start and join of processes
        with ProcessManager(process_objects=process_objects) as pm:
            queue_ins.put({'method': 'update_running_pv', 'data': True})
            for p, po in zip(pm.processes, pm.process_objects):
                main_logger.info(f"{str(po)} running on pid {p.pid} with parent pid {p._parent_pid}")

            while pm.is_running:
                time.sleep(1)

                alive_procs = [p.is_alive() for p in pm.processes]
                all_pm_procs_alive = all(alive_procs) and (len(pm.processes) > 0)

                for p, alive in zip(pm.processes, alive_procs):
                    if not alive:
                        main_logger.critical(f"Process {p} has stopped")

                if not all_pm_procs_alive:
                    main_logger.critical(f"Stopping all processes")
                elif not instrument_process.is_alive():
                    main_logger.critical(f"Instrumentation Process has stopped")
                    pm.is_running = False
                elif not logger_process.is_alive():
                    main_logger.critical(f"Logger Process has stopped")  # will not be logged
                    pm.is_running = False
                else:
                    main_logger.debug(f"All PM processes alive: {all_pm_procs_alive}")

            pipe_main.send(None)  # send end message to process_a
            for q in [queue_one, queue_two]:
                q.put(None)  # send end messages to processes b and c

            time.sleep(10)  # give the other processes time to stop

        if instrument_process.is_alive():
            queue_ins.put({'method': 'update_running_pv', 'data': False})

        queue_ins.put(None)
        instrument_process.join()  # wait for the instrument process to finish
        queue_log.put(None)
        logger_process.join()  # wait for the logger to finish last


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run with real or spoofed k2eg data.")
    parser.add_argument(
        "-skd",
        "--spoof_k2eg_data",
        action="store_true",
        help="Use spoofed k2eg data (random values) instead of real pv data.",
    )
    parser.add_argument(
        "-skda",
        "--spoof_k2eg_data_anomaly",
        action="store_true",
        help="Use spoofed k2eg data that always has anomalies, instead of real pv data.",
    )
    parser.add_argument(
        "-dfl",
        "--disable_file_logging",
        action="store_true",
        help=("Disable writing of log output to file, logging output might be output to terminal, " 
              "depending on that setting."),
    )
    parser.add_argument(
        "-dsl",
        "--disable_stdout_logging",
        action="store_true",
        help=("Disable writing of log output standard out, logging might still be sent to file, "
              "depending on that setting."),
    )
    parser.add_argument(
        "-ll",
        "--log_level",
        type=int,
        default=20,
        help="Level to perform logging at"
    )
    parser.add_argument(
        "-uk",
        "--use_k2eg",
        action="store_true",
        help="If this flag is passed, use k2eg to write to instrumentation PVs; otherwise use p4p"
    )

    args = parser.parse_args()

    main(args)
