SELECT source_posted_at, fraud_type
FROM signals_signal
WHERE (fraud_type = 'timepump' AND message_text NOT LIKE '%projects%' AND LENGTH(message_text) <= 800)
   OR fraud_type = 'crowdpump';
