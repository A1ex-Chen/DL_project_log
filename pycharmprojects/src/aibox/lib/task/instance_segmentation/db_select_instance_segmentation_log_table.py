def select_instance_segmentation_log_table(self) ->List[InstanceSegmentationLog
    ]:
    cursor = self._connection.cursor()
    instance_segmentation_logs = []
    for row in cursor.execute(DB.SQL_SELECT_INSTANCE_SEGMENTATION_LOG_TABLE):
        instance_segmentation_logs.append(DB.InstanceSegmentationLog(
            avg_anchor_objectness_loss=row[1], avg_anchor_transformer_loss=
            row[2], avg_proposal_class_loss=row[3],
            avg_proposal_transformer_loss=row[4], avg_mask_loss=row[5]))
    return instance_segmentation_logs
