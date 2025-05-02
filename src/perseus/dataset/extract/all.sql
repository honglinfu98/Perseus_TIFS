SELECT 
    p.*, -- This will select all columns from signals_pumpsignal
    s.telegram_chat_id, -- This will select telegram_chat_id from signals_signal
    s.commodity,
    s.source_posted_at,
    s.signal_position,
	s.message_text
FROM 
    signals_pumpsignal p
JOIN
    signals_signal s ON p.signal_id = s.id
WHERE 
    p.duration IS NOT NULL
AND s.message_text NOT ILIKE '%minute%'
AND s.message_text NOT ILIKE '%hour%'
AND s.message_text NOT ILIKE '%period%'
AND s.message_text NOT ILIKE '%until%'
AND s.message_text NOT ILIKE '%all%'
AND s.message_text NOT ILIKE '%mins%'
AND s.message_text NOT ILIKE '%time%'
AND s.message_text NOT ILIKE '%closed%'
AND LENGTH(s.message_text) >= 12
AND s.commodity IS NOT NULL
AND s.commodity != 'BNT'
AND s.commodity != 'RAMP'
AND s.commodity != 'SNM'
AND s.commodity != 'SNT'
AND s.commodity != 'SUPER'
AND s.commodity != 'WPR'
AND s.commodity != 'BIFI'
AND s.commodity != 'DREP'
AND s.commodity != 'EPS'
AND s.commodity != 'PIVX'
AND s.commodity != 'DLT'
AND p.pump_type = 'crowd'
AND p.pump_score != 0
AND s.source_posted_at::date <= '2024-02-16' 
ORDER BY s.source_posted_at DESC;
