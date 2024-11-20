import json
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from .task import Task


class DB:

    @dataclass
    class Log:

        class Status(Enum):
            INITIALIZING = 'initializing'
            INITIALIZED = 'initialized'
            RUNNING = 'running'
            STOPPED = 'stopped'
            FINISHED = 'finished'
            EXCEPTION = 'exception'

        @dataclass
        class Exception:
            code: str
            type: str
            message: str
            traceback: str

        global_batch: int
        status: Status
        datetime: int
        epoch: int
        total_epoch: int
        batch: int
        total_batch: int
        avg_loss: float
        learning_rate: float
        samples_per_sec: float
        eta_hrs: float
        exception: Optional[Exception] = None

        @staticmethod
        def serialize_exception(exception: Exception) -> str:
            return json.dumps({
                'code': exception.code,
                'type': exception.type,
                'message': exception.message,
                'traceback': exception.traceback,
            })

        @staticmethod
        def deserialize_exception(serialized_exception: str) -> Exception:
            exception_dict = json.loads(serialized_exception)
            return DB.Log.Exception(
                code=exception_dict['code'],
                type=exception_dict['type'],
                message=exception_dict['message'],
                traceback=exception_dict['traceback']
            )

    @dataclass
    class Checkpoint:

        @dataclass
        class Metrics:

            @dataclass
            class Overall:
                pass

            @dataclass
            class Specific:
                pass

            overall: Overall
            specific: Specific

        epoch: int
        avg_loss: float
        metrics: Metrics
        is_best: bool = False
        is_available: bool = True
        task_name: Task.Name = NotImplemented

        @staticmethod
        def serialize_metrics(metrics: Metrics) -> str:
            return json.dumps({
                'overall': metrics.overall.__dict__,
                'specific': metrics.specific.__dict__
            })

        @staticmethod
        def deserialize_metrics(serialized_metrics: str) -> Metrics:
            metric_dict = json.loads(serialized_metrics)
            return DB.Checkpoint.Metrics(**metric_dict)

    SQL_CREATE_LOG_TABLE = '''
        CREATE TABLE IF NOT EXISTS log(
            sn INTEGER PRIMARY KEY AUTOINCREMENT,
            global_batch INT NOT NULL,
            status TEXT NOT NULL,
            datetime DATETIME NOT NULL,
            epoch INT NOT NULL,
            total_epoch INT NOT NULL,
            batch INT NOT NULL,
            total_batch INT NOT NULL,
            avg_loss REAL NOT NULL,
            learning_rate REAL NOT NULL,
            samples_per_sec REAL NOT NULL,
            eta_hrs REAL NOT NULL,
            exception TEXT
        );
    '''

    SQL_INSERT_LOG_TABLE = '''
        INSERT INTO log (global_batch, status, datetime, epoch, total_epoch, batch, total_batch, avg_loss, learning_rate, samples_per_sec, eta_hrs, exception)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    '''

    SQL_SELECT_LOG_TABLE_LATEST = '''
        SELECT * FROM log ORDER BY sn DESC LIMIT 1;
    '''

    SQL_UPDATE_LOG_TABLE_LATEST_STATUS = '''
        UPDATE log SET datetime = ?, status = ? WHERE sn = (SELECT MAX(sn) FROM log);
    '''

    SQL_UPDATE_LOG_TABLE_LATEST_EXCEPTION = '''
        UPDATE log SET datetime = ?, status = ?, exception = ? WHERE sn = (SELECT MAX(sn) FROM log);
    '''

    SQL_SELECT_LOG_TABLE = '''
        SELECT * FROM log;
    '''

    SQL_CREATE_CHECKPOINT_TABLE = '''
        CREATE TABLE IF NOT EXISTS checkpoint(
            epoch INT PRIMARY KEY NOT NULL,
            avg_loss REAL NOT NULL,
            metrics TEXT,
            is_best BOOLEAN,
            is_available BOOLEAN,
            task_name TEXT
        );
    '''

    SQL_INSERT_CHECKPOINT_TABLE = '''
        INSERT INTO checkpoint (epoch, avg_loss, metrics, is_best, is_available, task_name)
        VALUES (?, ?, ?, ?, ?, ?);
    '''

    SQL_UPDATE_CHECKPOINT_TABLE_IS_BEST_FOR_EPOCH = '''
        UPDATE checkpoint SET is_best = ? WHERE epoch = ?;
    '''

    SQL_UPDATE_CHECKPOINT_TABLE_IS_AVAILABLE_FOR_EPOCH = '''
        UPDATE checkpoint SET is_available = ? WHERE epoch = ?;
    '''

    SQL_SELECT_CHECKPOINT_TABLE = '''
        SELECT * FROM checkpoint;
    '''

    SQL_SELECT_CHECKPOINT_TABLE_FOR_EPOCH = '''
        SELECT * FROM checkpoint WHERE epoch = ?;
    '''

    SQL_SELECT_CHECKPOINT_TABLE_LATEST = '''
        SELECT * FROM checkpoint WHERE epoch = (SELECT MAX(epoch) FROM checkpoint);
    '''














        @staticmethod

    @dataclass
    class Checkpoint:

        @dataclass
        class Metrics:

            @dataclass
            class Overall:
                pass

            @dataclass
            class Specific:
                pass

            overall: Overall
            specific: Specific

        epoch: int
        avg_loss: float
        metrics: Metrics
        is_best: bool = False
        is_available: bool = True
        task_name: Task.Name = NotImplemented

        @staticmethod

        @staticmethod

    SQL_CREATE_LOG_TABLE = '''
        CREATE TABLE IF NOT EXISTS log(
            sn INTEGER PRIMARY KEY AUTOINCREMENT,
            global_batch INT NOT NULL,
            status TEXT NOT NULL,
            datetime DATETIME NOT NULL,
            epoch INT NOT NULL,
            total_epoch INT NOT NULL,
            batch INT NOT NULL,
            total_batch INT NOT NULL,
            avg_loss REAL NOT NULL,
            learning_rate REAL NOT NULL,
            samples_per_sec REAL NOT NULL,
            eta_hrs REAL NOT NULL,
            exception TEXT
        );
    '''

    SQL_INSERT_LOG_TABLE = '''
        INSERT INTO log (global_batch, status, datetime, epoch, total_epoch, batch, total_batch, avg_loss, learning_rate, samples_per_sec, eta_hrs, exception)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    '''

    SQL_SELECT_LOG_TABLE_LATEST = '''
        SELECT * FROM log ORDER BY sn DESC LIMIT 1;
    '''

    SQL_UPDATE_LOG_TABLE_LATEST_STATUS = '''
        UPDATE log SET datetime = ?, status = ? WHERE sn = (SELECT MAX(sn) FROM log);
    '''

    SQL_UPDATE_LOG_TABLE_LATEST_EXCEPTION = '''
        UPDATE log SET datetime = ?, status = ?, exception = ? WHERE sn = (SELECT MAX(sn) FROM log);
    '''

    SQL_SELECT_LOG_TABLE = '''
        SELECT * FROM log;
    '''

    SQL_CREATE_CHECKPOINT_TABLE = '''
        CREATE TABLE IF NOT EXISTS checkpoint(
            epoch INT PRIMARY KEY NOT NULL,
            avg_loss REAL NOT NULL,
            metrics TEXT,
            is_best BOOLEAN,
            is_available BOOLEAN,
            task_name TEXT
        );
    '''

    SQL_INSERT_CHECKPOINT_TABLE = '''
        INSERT INTO checkpoint (epoch, avg_loss, metrics, is_best, is_available, task_name)
        VALUES (?, ?, ?, ?, ?, ?);
    '''

    SQL_UPDATE_CHECKPOINT_TABLE_IS_BEST_FOR_EPOCH = '''
        UPDATE checkpoint SET is_best = ? WHERE epoch = ?;
    '''

    SQL_UPDATE_CHECKPOINT_TABLE_IS_AVAILABLE_FOR_EPOCH = '''
        UPDATE checkpoint SET is_available = ? WHERE epoch = ?;
    '''

    SQL_SELECT_CHECKPOINT_TABLE = '''
        SELECT * FROM checkpoint;
    '''

    SQL_SELECT_CHECKPOINT_TABLE_FOR_EPOCH = '''
        SELECT * FROM checkpoint WHERE epoch = ?;
    '''

    SQL_SELECT_CHECKPOINT_TABLE_LATEST = '''
        SELECT * FROM checkpoint WHERE epoch = (SELECT MAX(epoch) FROM checkpoint);
    '''

    def __init__(self, path_to_db: str):
        super().__init__()
        self._connection = sqlite3.connect(path_to_db)
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_CREATE_LOG_TABLE)
        cursor.execute(DB.SQL_CREATE_CHECKPOINT_TABLE)
        self._connection.commit()

    def insert_log_table(self, log: Log):
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_INSERT_LOG_TABLE, (log.global_batch, log.status.value, log.datetime,
                                                 log.epoch, log.total_epoch,
                                                 log.batch, log.total_batch,
                                                 log.avg_loss,
                                                 log.learning_rate, log.samples_per_sec, log.eta_hrs,
                                                 log.exception if log.exception is None else DB.Log.serialize_exception(log.exception)))
        self._connection.commit()

    def select_log_table_latest(self) -> Log:
        cursor = self._connection.cursor()
        row = next(cursor.execute(DB.SQL_SELECT_LOG_TABLE_LATEST))
        log = DB.Log(
            global_batch=row[1],
            status=DB.Log.Status(row[2]),
            datetime=row[3],
            epoch=row[4],
            total_epoch=row[5],
            batch=row[6],
            total_batch=row[7],
            avg_loss=row[8],
            learning_rate=row[9],
            samples_per_sec=row[10],
            eta_hrs=row[11],
            exception=row[12] if row[12] is None else DB.Log.deserialize_exception(row[12])
        )
        return log

    def update_log_table_latest_status(self, status: Log.Status):
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_UPDATE_LOG_TABLE_LATEST_STATUS, (int(time.time()), status.value))
        self._connection.commit()

    def update_log_table_latest_exception(self, exception: Log.Exception):
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_UPDATE_LOG_TABLE_LATEST_EXCEPTION, (int(time.time()), DB.Log.Status.EXCEPTION.value,
                                                                  DB.Log.serialize_exception(exception)))
        self._connection.commit()

    def select_log_table(self) -> List[Log]:
        cursor = self._connection.cursor()
        logs = []
        for row in cursor.execute(DB.SQL_SELECT_LOG_TABLE):
            logs.append(DB.Log(
                global_batch=row[1],
                status=DB.Log.Status(row[2]),
                datetime=row[3],
                epoch=row[4],
                total_epoch=row[5],
                batch=row[6],
                total_batch=row[7],
                avg_loss=row[8],
                learning_rate=row[9],
                samples_per_sec=row[10],
                eta_hrs=row[11],
                exception=row[12] if row[12] is None else DB.Log.deserialize_exception(row[12])
            ))
        return logs

    def insert_checkpoint_table(self, checkpoint: Checkpoint):
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_INSERT_CHECKPOINT_TABLE, (checkpoint.epoch,
                                                        checkpoint.avg_loss,
                                                        self.Checkpoint.serialize_metrics(checkpoint.metrics),
                                                        checkpoint.is_best,
                                                        checkpoint.is_available,
                                                        checkpoint.task_name.value))
        self._connection.commit()

    def update_checkpoint_table_is_best_for_epoch(self, is_best: bool, epoch: int):
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_UPDATE_CHECKPOINT_TABLE_IS_BEST_FOR_EPOCH, (is_best, epoch))
        self._connection.commit()

    def update_checkpoint_table_is_available_for_epoch(self, is_available: bool, epoch: int):
        cursor = self._connection.cursor()
        cursor.execute(DB.SQL_UPDATE_CHECKPOINT_TABLE_IS_AVAILABLE_FOR_EPOCH, (is_available, epoch))
        self._connection.commit()

    def select_checkpoint_table(self) -> List[Checkpoint]:
        cursor = self._connection.cursor()
        checkpoints = []
        for row in cursor.execute(DB.SQL_SELECT_CHECKPOINT_TABLE):
            checkpoints.append(DB.Checkpoint(
                epoch=row[0],
                avg_loss=row[1],
                metrics=self.Checkpoint.deserialize_metrics(row[2]),
                is_best=row[3],
                is_available=row[4],
                task_name=Task.Name(row[5])
            ))
        return checkpoints

    def select_checkpoint_table_for_epoch(self, epoch: int) -> Checkpoint:
        cursor = self._connection.cursor()
        row = next(cursor.execute(DB.SQL_SELECT_CHECKPOINT_TABLE_FOR_EPOCH, (epoch,)))
        checkpoint = DB.Checkpoint(
            epoch=row[0],
            avg_loss=row[1],
            metrics=self.Checkpoint.deserialize_metrics(row[2]),
            is_best=row[3],
            is_available=row[4],
            task_name=Task.Name(row[5])
        )
        return checkpoint

    def select_checkpoint_table_latest(self) -> Optional[Checkpoint]:
        cursor = self._connection.cursor()
        row = next(cursor.execute(DB.SQL_SELECT_CHECKPOINT_TABLE_LATEST), None)
        if row:
            checkpoint = DB.Checkpoint(
                epoch=row[0],
                avg_loss=row[1],
                metrics=self.Checkpoint.deserialize_metrics(row[2]),
                is_best=row[3],
                is_available=row[4],
                task_name=Task.Name(row[5])
            )
        else:
            checkpoint = None
        return checkpoint

    def close(self):
        self._connection.close()