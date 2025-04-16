def insert_detection_log_table(self, detection_log: DetectionLog):
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_INSERT_DETECTION_LOG_TABLE, (detection_log.
        avg_anchor_objectness_loss, detection_log.
        avg_anchor_transformer_loss, detection_log.avg_proposal_class_loss,
        detection_log.avg_proposal_transformer_loss))
    self._connection.commit()
