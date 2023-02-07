WITH avg_scores AS (
SELECT
entity_id,
AVG(CASE WHEN channel_time_score IS NOT NULL then (channel_time_score) ELSE 0 END) AS time_score,
AVG(CASE WHEN channel_crowd_score IS NOT NULL then (channel_crowd_score) ELSE 0 END) AS crowd_score
FROM cloudburst_signals
GROUP BY entity_id
),
ind_scores AS (
SELECT
cm."user_PID" as user_pid,
SUM(CASE WHEN Lower(cm.flair) = 'administrator' THEN 0.5 ELSE 0 END) AS score_admin,
SUM(CASE WHEN Lower(cm.flair) = 'owner' THEN 0.25 ELSE 0 END) AS score_owner,
SUM(CASE WHEN Lower(cm.flair) = 'member' THEN 0.1 ELSE 0 END) AS score_member,
sum(sc.time_score) as time_score,
sum(sc.crowd_score) as crowd_score
FROM
telegram_chats_members cm
JOIN telegram_chats tc ON cm."chat_PID" = tc.pid
JOIN avg_scores sc ON tc.entity_id = sc.entity_id
GROUP BY 1
)
SELECT
user_pid,
score_admin,
score_owner,
score_member,
time_score,
crowd_score,
(score_admin + score_owner + score_member + time_score + crowd_score) AS total_score
FROM ind_scores
ORDER BY 7 DESC