"""
调度工作进程
使用 multiprocessing.Process 实现独立的调度计算进程
通过 Queue 进行跨进程通信，异步处理调度请求
"""

import multiprocessing
from multiprocessing import Queue, Process
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time
import uuid

from .types import (
    SolverConfig,
    DeviceState,
    UrgentBatch,
    ScheduledBatch,
)
from .solver import IntegerProgrammingSolver


class SchedulingWorker:
    """
    调度工作进程
    在独立进程中运行调度算法，通过队列异步处理请求

    支持的任务类型：
    - 'solve': 求解调度问题
    - 'reschedule': 紧急插单重调度
    - 'validate': 验证调度计划
    - 'stop': 停止工作进程
    """

    def __init__(self, config: SolverConfig):
        self.config = config
        self._request_queue: Queue = Queue()
        self._result_queue: Queue = Queue()
        self._process: Optional[Process] = None
        self._running = False
        self._active_tasks: Dict[str, Dict[str, Any]] = {}

    def start(self) -> None:
        """
        启动工作进程
        """
        if self._process is not None and self._process.is_alive():
            return

        self._running = True
        self._process = Process(
            target=self._worker_loop,
            args=(self.config, self._request_queue, self._result_queue),
            daemon=True
        )
        self._process.start()
        print(f"[SchedulingWorker] 工作进程已启动，PID: {self._process.pid}")

    def stop(self, timeout: float = 5.0) -> None:
        """
        停止工作进程

        Args:
            timeout: 等待进程结束的超时时间（秒）
        """
        if self._process is None or not self._process.is_alive():
            self._running = False
            return

        self._running = False

        try:
            self._request_queue.put({'task_id': 'internal_stop', 'task_type': 'stop', 'params': {}})
        except Exception:
            pass

        if self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                print(f"[SchedulingWorker] 进程未正常退出，强制终止")
                self._process.terminate()
                self._process.join(timeout=2.0)

        print("[SchedulingWorker] 工作进程已停止")
        self._active_tasks.clear()

    def is_alive(self) -> bool:
        """
        检查工作进程是否存活

        Returns:
            bool: 进程是否存活
        """
        return self._process is not None and self._process.is_alive()

    def submit_solve(self,
                     device_states: Dict[int, DeviceState],
                     required_batches: int,
                     time_horizon_hours: int = 24,
                     start_time: Optional[datetime] = None) -> str:
        """
        提交调度求解任务

        Args:
            device_states: 设备状态字典
            required_batches: 需要调度的批次数量
            time_horizon_hours: 调度时间范围（小时）
            start_time: 调度开始时间

        Returns:
            str: 任务ID，用于查询结果
        """
        task_id = str(uuid.uuid4())
        task = {
            'task_id': task_id,
            'task_type': 'solve',
            'params': {
                'device_states': device_states,
                'required_batches': required_batches,
                'time_horizon_hours': time_horizon_hours,
                'start_time': start_time,
            }
        }
        self._request_queue.put(task)
        self._active_tasks[task_id] = {
            'task_type': 'solve',
            'submitted_at': time.time(),
            'completed': False,
        }
        return task_id

    def submit_reschedule(self,
                          current_schedule: List[ScheduledBatch],
                          device_states: Dict[int, DeviceState],
                          urgent_batch: UrgentBatch,
                          time_horizon_hours: int = 24,
                          current_time: Optional[datetime] = None) -> str:
        """
        提交紧急插单重调度任务

        Args:
            current_schedule: 当前调度计划
            device_states: 设备状态字典
            urgent_batch: 紧急批次信息
            time_horizon_hours: 调度时间范围（小时）
            current_time: 当前时间

        Returns:
            str: 任务ID，用于查询结果
        """
        task_id = str(uuid.uuid4())
        task = {
            'task_id': task_id,
            'task_type': 'reschedule',
            'params': {
                'current_schedule': current_schedule,
                'device_states': device_states,
                'urgent_batch': urgent_batch,
                'time_horizon_hours': time_horizon_hours,
                'current_time': current_time,
            }
        }
        self._request_queue.put(task)
        self._active_tasks[task_id] = {
            'task_type': 'reschedule',
            'submitted_at': time.time(),
            'completed': False,
        }
        return task_id

    def submit_validate(self,
                        schedule: List[ScheduledBatch],
                        time_horizon_hours: int,
                        start_time: datetime) -> str:
        """
        提交调度验证任务

        Args:
            schedule: 调度计划
            time_horizon_hours: 时间范围（小时）
            start_time: 开始时间

        Returns:
            str: 任务ID，用于查询结果
        """
        task_id = str(uuid.uuid4())
        task = {
            'task_id': task_id,
            'task_type': 'validate',
            'params': {
                'schedule': schedule,
                'time_horizon_hours': time_horizon_hours,
                'start_time': start_time,
            }
        }
        self._request_queue.put(task)
        self._active_tasks[task_id] = {
            'task_type': 'validate',
            'submitted_at': time.time(),
            'completed': False,
        }
        return task_id

    def get_result(self, task_id: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        获取任务结果

        Args:
            task_id: 任务ID
            timeout: 等待超时时间（秒），0表示不等待

        Returns:
            Optional[Dict]: 结果字典，包含：
                - 'task_id': 任务ID
                - 'success': 是否成功
                - 'result': 结果数据（成功时）
                - 'error': 错误信息（失败时）
            如果结果尚未就绪，返回 None
        """
        if task_id not in self._active_tasks:
            return None

        if self._active_tasks[task_id]['completed']:
            return self._active_tasks[task_id].get('result')

        try:
            if timeout > 0:
                result = self._result_queue.get(timeout=timeout)
            else:
                if self._result_queue.empty():
                    return None
                result = self._result_queue.get_nowait()

            if result['task_id'] in self._active_tasks:
                self._active_tasks[result['task_id']]['completed'] = True
                self._active_tasks[result['task_id']]['result'] = result
                self._active_tasks[result['task_id']]['completed_at'] = time.time()

            if result['task_id'] == task_id:
                return result
            else:
                return None
        except Exception:
            return None

    def get_all_results(self, timeout: float = 0.0) -> List[Dict[str, Any]]:
        """
        获取所有就绪的结果

        Args:
            timeout: 等待超时时间（秒）

        Returns:
            List[Dict]: 结果字典列表
        """
        results = []
        try:
            while True:
                if timeout > 0:
                    result = self._result_queue.get(timeout=timeout)
                    timeout = 0.0
                else:
                    if self._result_queue.empty():
                        break
                    result = self._result_queue.get_nowait()

                if result['task_id'] in self._active_tasks:
                    self._active_tasks[result['task_id']]['completed'] = True
                    self._active_tasks[result['task_id']]['result'] = result
                    self._active_tasks[result['task_id']]['completed_at'] = time.time()

                results.append(result)
        except Exception:
            pass
        return results

    def cleanup_completed_tasks(self, max_age_seconds: float = 300.0) -> int:
        """
        清理已完成的任务记录

        Args:
            max_age_seconds: 保留已完成任务的最大时间（秒）

        Returns:
            int: 清理的任务数量
        """
        now = time.time()
        to_remove = []
        for task_id, info in self._active_tasks.items():
            if info['completed'] and 'completed_at' in info:
                if now - info['completed_at'] > max_age_seconds:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._active_tasks[task_id]

        return len(to_remove)

    def get_queue_sizes(self) -> Tuple[int, int]:
        """
        获取队列大小

        Returns:
            Tuple[int, int]: (请求队列大小, 结果队列大小)
        """
        try:
            req_size = self._request_queue.qsize()
        except NotImplementedError:
            req_size = -1

        try:
            res_size = self._result_queue.qsize()
        except NotImplementedError:
            res_size = -1

        return req_size, res_size

    @staticmethod
    def _worker_loop(config: SolverConfig,
                     request_queue: Queue,
                     result_queue: Queue) -> None:
        """
        工作进程主循环
        在独立进程中运行，处理调度请求

        Args:
            config: 求解器配置
            request_queue: 请求队列
            result_queue: 结果队列
        """
        print("[SchedulingWorker] 工作进程初始化...")

        try:
            solver = IntegerProgrammingSolver(config)
            print(f"[SchedulingWorker] 求解器初始化完成，使用: {'PuLP' if solver._has_pulp else 'Heuristic'}")
        except Exception as e:
            print(f"[SchedulingWorker] 求解器初始化失败: {e}")
            result_queue.put({
                'task_id': 'init_error',
                'success': False,
                'result': None,
                'error': f"Solver initialization failed: {e}"
            })
            return

        running = True

        while running:
            try:
                task = request_queue.get(timeout=1.0)
                task_id = task['task_id']
                task_type = task['task_type']
                params = task['params']

                if task_type == 'stop':
                    print("[SchedulingWorker] 收到停止信号")
                    running = False
                    continue

                try:
                    start_time = time.time()

                    if task_type == 'solve':
                        result = solver.solve(
                            device_states=params['device_states'],
                            required_batches=params['required_batches'],
                            time_horizon_hours=params['time_horizon_hours'],
                            start_time=params['start_time'],
                        )
                        result_queue.put({
                            'task_id': task_id,
                            'success': True,
                            'result': {
                                'schedule': result[0],
                                'total_cost': result[1],
                                'energy_saving': result[2],
                                'status': result[3],
                            },
                            'error': None,
                            'duration_ms': int((time.time() - start_time) * 1000),
                        })

                    elif task_type == 'reschedule':
                        result = solver.reschedule_for_urgent_batch(
                            current_schedule=params['current_schedule'],
                            device_states=params['device_states'],
                            urgent_batch=params['urgent_batch'],
                            time_horizon_hours=params['time_horizon_hours'],
                            current_time=params['current_time'],
                        )
                        result_queue.put({
                            'task_id': task_id,
                            'success': True,
                            'result': {
                                'new_schedule': result[0],
                                'urgent_cost': result[1],
                                'cost_delta': result[2],
                                'status': result[3],
                            },
                            'error': None,
                            'duration_ms': int((time.time() - start_time) * 1000),
                        })

                    elif task_type == 'validate':
                        result = solver.validate_schedule(
                            schedule=params['schedule'],
                            time_horizon_hours=params['time_horizon_hours'],
                            start_time=params['start_time'],
                        )
                        result_queue.put({
                            'task_id': task_id,
                            'success': True,
                            'result': {
                                'is_valid': result[0],
                                'violations': result[1],
                            },
                            'error': None,
                            'duration_ms': int((time.time() - start_time) * 1000),
                        })

                    else:
                        result_queue.put({
                            'task_id': task_id,
                            'success': False,
                            'result': None,
                            'error': f"Unknown task type: {task_type}",
                            'duration_ms': 0,
                        })

                except Exception as e:
                    print(f"[SchedulingWorker] 任务执行失败 [{task_id}]: {e}")
                    result_queue.put({
                        'task_id': task_id,
                        'success': False,
                        'result': None,
                        'error': str(e),
                        'duration_ms': 0,
                    })

            except multiprocessing.queues.Empty:
                continue
            except Exception as e:
                print(f"[SchedulingWorker] 主循环异常: {e}")
                continue

        print("[SchedulingWorker] 工作进程退出")

    def __enter__(self) -> 'SchedulingWorker':
        """
        上下文管理器入口
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        上下文管理器出口
        """
        self.stop()
