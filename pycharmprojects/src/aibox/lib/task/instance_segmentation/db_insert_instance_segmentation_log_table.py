def insert_instance_segmentation_log_table(self, instance_segmentation_log:
    InstanceSegmentationLog):
    cursor = self._connection.cursor()
    cursor.execute(DB.SQL_INSERT_INSTANCE_SEGMENTATION_LOG_TABLE, (
        instance_segmentation_log.avg_anchor_objectness_loss,
        instance_segmentation_log.avg_anchor_transformer_loss,
        instance_segmentation_log.avg_proposal_class_loss,
        instance_segmentation_log.avg_proposal_transformer_loss,
        instance_segmentation_log.avg_mask_loss))
    self._connection.commit()
