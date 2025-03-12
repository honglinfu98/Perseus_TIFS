SELECT 
    entity_id,
    SUM(CASE WHEN fraud_type = 'crowdpump' THEN 1 ELSE 0 END) AS crowd_pump_count,
    SUM(CASE WHEN fraud_type = 'timepump' THEN 1 ELSE 0 END) AS time_pump_count
FROM signals_signal
WHERE 
    (fraud_type = 'timepump' AND message_text NOT LIKE '%projects%' AND LENGTH(message_text) <= 800)
    OR fraud_type = 'crowdpump'
GROUP BY entity_id
ORDER BY entity_id;
