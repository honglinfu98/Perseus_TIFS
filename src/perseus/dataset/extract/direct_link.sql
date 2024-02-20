SELECT
    c1.chat_id AS chat_id_1,
    c2.chat_id AS chat_id_2,
    c1.user_id AS shared_user_id
FROM
    community_data_telegramchatmember c1
JOIN
    community_data_telegramchatmember c2 ON c1.user_id = c2.user_id AND c1.chat_id <> c2.chat_id
WHERE
    (LOWER(c1.flair) = 'administrator' OR LOWER(c1.flair) = 'owner')
    AND (LOWER(c2.flair) = 'administrator' OR LOWER(c2.flair) = 'owner')
GROUP BY
    c1.chat_id, c2.chat_id, c1.user_id
ORDER BY
    c1.chat_id, c2.chat_id;
