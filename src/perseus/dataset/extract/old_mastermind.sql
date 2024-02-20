WITH RankedSignals AS (
    SELECT
        signals.id as signal_id,
        source_posted_at,
        commodity,
		base_commodity,
        telegram_chat_id,
        trade_type,
        LAG(source_posted_at) OVER (PARTITION BY commodity ORDER BY source_posted_at) AS prev_posted_at
    FROM signals_signal as signals
	LEFT JOIN signals_pumpsignal as scores ON signals.id = scores.signal_id
    WHERE trade_type = 'buy'
    AND commodity IS NOT NULL
    AND commodity != '/'
	AND commodity != '$'
	AND commodity != '('
	AND pump_score > 0
	AND pump_score IS NOT NULL
    AND source_posted_at >= '2017-10-01' -- Start date
    AND source_posted_at <= '2022-10-01' -- End date
),
GapsIdentified AS (
    SELECT
        *,
        CASE
            WHEN EXTRACT(EPOCH FROM (source_posted_at - prev_posted_at)) / 3600 > 24 THEN 1
            ELSE 0
        END AS isNewEvent
    FROM RankedSignals
),
EventMarkers AS (
    SELECT
        *,
        SUM(isNewEvent) OVER (PARTITION BY commodity ORDER BY source_posted_at) AS event_id
    FROM GapsIdentified
),
EventAggregations AS (
    SELECT
        event_id,
        commodity,
        ARRAY_AGG(signal_id ORDER BY source_posted_at) AS signal_ids,
        COUNT(*) AS signal_count
    FROM EventMarkers
    GROUP BY event_id, commodity
),
FirstSignalInEvent AS (
    SELECT
        event_id,
        commodity,
        FIRST_VALUE(signal_id) OVER (PARTITION BY event_id, commodity ORDER BY source_posted_at) AS first_signal_id
    FROM EventMarkers
)
SELECT DISTINCT
    E.signal_id,
    E.source_posted_at,
    E.commodity,
	E.base_commodity,
    E.telegram_chat_id,
    E.event_id,
    A.signal_ids,
    A.signal_count
FROM EventMarkers E
JOIN EventAggregations A ON E.event_id = A.event_id AND E.commodity = A.commodity
JOIN FirstSignalInEvent F ON E.event_id = F.event_id AND E.commodity = F.commodity AND E.signal_id = F.first_signal_id
ORDER BY E.commodity, E.source_posted_at, A.signal_count DESC;
