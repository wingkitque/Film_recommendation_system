# 第一步：收集数据
# https://grouplens.org/datasets/movielens/
# ->  recommended for education and development
# ->  ml-latest-small.zip (size: 1 MB)

# 第二步：准备数据
import pandas as pd
import numpy as np
import tensorflow as tf

# ratings_df 含有：userId（用户id）,movieId（电影id）,rating（评分）,timestamp（时间戳）
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')

# movies_df 含有：movieId（电影id）,title（电影标题）,genres（电影类型）
movies_df = pd.read_csv('ml-latest-small/movies.csv')
# 为 movies_df 表数据增加行索引 movieRow （从0开始增加）
movies_df['movieRow'] = movies_df.index

# 筛选movie_df 中的特征，去掉 列数据 genres（电影类型）
movies_df = movies_df[['movieRow', 'movieId', 'title']]
# 保存过程文件moviesProcessed.csv      此时 movies_df 表含有：movieRow（行索引）,movieId（电影id）,title（电影标题）
movies_df.to_csv('moviesProcessed.csv', index=False, header=True, encoding='utf-8')

# 将ratings_df中的movies替换为行号，通过 movieId 将 ratings_df 和 movies_df 两表合并成一张大表
ratings_df = pd.merge(ratings_df, movies_df, on='movieId')

# 筛选ratings_df 中的特征
ratings_df = ratings_df[['userId', 'movieRow', 'rating']]
# 保存过程文件ratingsProcessed.csv      此时 ratings_df 表含有：userId（用户id）,movieRow（行索引）,rating（电影评分）
ratings_df.to_csv('ratingsProcessed.csv', index=False, header=True, encoding='utf-8')

# 创建电影评分矩阵 rating 和评分记录矩阵 recode
# 获取两表的 行数
userNo = ratings_df['userId'].max()+1
movieNo = ratings_df['movieRow'].max()+1

rating = np.zeros((movieNo, userNo))
flag = 0
ratings_df_length = np.shape(ratings_df)[0]
for index, row in ratings_df.iterrows():
    rating[int(row['movieRow']), int(row['userId'])] = row['rating']
    flag += 1
    # print('processed %d, %d left' % (flag, ratings_df_length-flag))

# 评分记录表：有评分的项为'1'，没评分的项为'0'
record = rating > 0
record = np.array(record, dtype=int)


# 第三步：构建模型
def normalizeRatings(rating, record):
    m, n = rating.shape
    rating_mean = np.zeros((m, 1))  # 将所有电影评分初始化为0
    rating_norm = np.zeros((m, n))  # 保存处理后的数据
    # 将原始评分减去平均评分
    for i in range(m):
        idx = record[i, :] != 0
        rating_mean[i] = np.mean(rating[i, idx])
        rating_norm[i, idx] -= rating_mean[i]
    # 将计算结果和平均评分返回   rating_mean 为多行一列的表，内容为电影平均分；rating_norm 为 一张0和负数的表
    return rating_norm, rating_mean



if __name__ == '__main__':
    rating_norm, rating_mean = normalizeRatings(rating, record)
    rating_norm = np.nan_to_num(rating_norm)
    rating_mean = np.nan_to_num(rating_mean)
    # print(rating_norm)
    # print(rating_mean)

    num_features = 10
    X_parameters = tf.Variable(tf.random_normal([movieNo, num_features], stddev=0.35))
    Theta_paramters = tf.Variable(tf.random_normal([userNo, num_features], stddev=0.35))
    loss = 1 / 2 * tf.reduce_sum(
        ((tf.matmul(X_parameters, Theta_paramters, transpose_b=True) - rating_norm) * record) ** 2) + 1 / 2 * (
                       tf.reduce_sum(X_parameters ** 2) + tf.reduce_sum(Theta_paramters ** 2))

    optimizer = tf.train.AdamOptimizer(1e-4)
    train = optimizer.minimize(loss)

# 第四步：训练模型
    tf.summary.scalar('loss', loss)
    summaryMerged = tf.summary.merge_all()
    filename = './movie_tensorboad'
    writer = tf.summary.FileWriter(filename)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    n = 5000
    for i in range(n):
        _, movie_summary = sess.run([train, summaryMerged])
        writer.add_summary(movie_summary, i)
        if i % 100 == 0:
            print('第 ', i, '次训练，还有',  n-i, '次~~')

# 第五步：评估模型
Current_X_parameters, Current_Theta_parameters = sess.run([X_parameters, Theta_paramters])
predicts = np.dot(Current_X_parameters, Current_Theta_parameters.T) + rating_mean
errors = np.sqrt(np.sum((predicts - rating) ** 2))
print(errors)

# 第六步：构建完整的电影评分系统
user_id = input('您要向哪位用户进行推荐？请输入用户编号：')
sortedResult = predicts[:, int(user_id)].argsort()[::-1]
idx = 0
print('为该用户推荐的评分最高的20部电影是：'.center(80, '='))
for i in sortedResult:
    print('评分：%.2f，电影名：%s' % (predicts[i, int(user_id)], movies_df.iloc[i]['title']))
    idx += 1
    if idx == 20:
        break
