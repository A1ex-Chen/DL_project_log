def select_detection_log_table(self) ->List[DetectionLog]:
    cursor = self._connection.cursor()
    detection_logs = []
    for row in cursor.execute(DB.SQL_SELECT_DETECTION_LOG_TABLE):
        detection_logs.append(DB.DetectionLog(avg_anchor_objectness_loss=
            row[1], avg_anchor_transformer_loss=row[2],
            avg_proposal_class_loss=row[3], avg_proposal_transformer_loss=
            row[4]))
    return detection_logs
