# -*- coding: utf-8 -*-

import sqlite3


# 别忘记用完后关闭游标和连接
class SqlLiteManage:
    def __init__(self):
        pass

    # 连接数据库， 返回游标
    def connect_db(self):
        connect = sqlite3.connect('E:/Opencv/FaceRecognition_demo/database/data.db')
        cursor = connect.cursor()
        return connect, cursor

    # 初始化数据库
    def root_db(self):
        sql = '''
        DELETE 
        FROM face_data_set
        '''
        connect, cursor = self.connect_db()
        cursor.execute(sql)
        connect.commit()
        cursor.close()
        connect.close()

    # 建表
    def create_table(self):
        sql = '''
        CREATE TABLE IF NOT EXISTS face_data_set (
            FACE_ID INT PRIMARY KEY NOT NULL ,
            USR_NAME VARCHAR(255) NOT NULL,
            USR_NAME_CHAR VARCHAR(255) NOT NULL,
            FACE_IS_TRAINED INT NOT NULL
        )
        '''
        connect, cursor = self.connect_db()
        cursor.execute(sql)
        connect.commit()
        cursor.close()
        connect.close()

    # 删表
    def delete_table(self):
        sql = '''
        DROP TABLE IF EXISTS face_data_set
        '''
        connect, cursor = self.connect_db()
        cursor.execute(sql)
        connect.commit()
        cursor.close()
        connect.close()

    # 查询
    def search(self, field, value=None):
        isChecked = True
        sql = ''
        if field == 'FACE_ID':
            sql = '''
            SELECT * FROM face_data_set WHERE FACE_ID = ?
            '''
        elif field == 'USR_NAME' and value != None:
            sql = '''
                SELECT * FROM face_data_set WHERE USR_NAME = ?
            '''
        elif field == 'USR_NAME' and value == None:
            sql = '''
                SELECT USR_NAME FROM face_data_set
            '''
        try:
            connect, cursor = self.connect_db()
            if value:
                cursor.execute(sql, (value,))
                data = cursor.fetchall()[-1]
                connect.commit()
                cursor.close()
                connect.close()
                return isChecked, data
            else:
                cursor.execute(sql)
                names = cursor.fetchall()[-1]
                connect.commit()
                cursor.close()
                connect.close()
                return names
        except:
            isChecked = False
            return isChecked, ''

    # 插入
    def insert(self, item):
        sql = '''
        INSERT INTO face_data_set VALUES (?, ?, ?, ?)
        '''
        connect, cursor = self.connect_db()
        cursor.execute(sql, (item[0], item[1], item[2], item[3],))
        connect.commit()
        cursor.close()
        connect.close()

    # 删除
    # 先查找然后返回，确认存在后再删除
    def remove(self, field, value):
        sql = ''
        sql2 = ''
        if field == 'FACE_ID':
            sql = '''
                    DELETE FROM face_data_set WHERE FACE_ID = ?
            '''
            sql2 = '''
            SELECT * FROM face_data_set WHERE FACE_ID = ?
            '''
        elif field == 'USR_NAME':
            sql = '''
                        DELETE FROM face_data_set WHERE USR_NAME = ?
            '''
            sql2 = '''
                        SELECT * FROM face_data_set WHERE USR_NAME = ?
            '''
        connect, cursor = self.connect_db()
        try:
            cursor.execute(sql2, (value,))
            data = cursor.fetchall()[-1]
            if not data:
                return ''
            else:
                cursor.execute(sql, (value,))
                connect.commit()
                cursor.close()
                connect.close()
                return data
        except:
            cursor.close()
            connect.close()
            return ''

    # 数据数量计数
    def count_data(self):
        sql = '''
        SELECT COUNT(*) FROM face_data_set
        '''
        connect, cursor = self.connect_db()
        cursor.execute(sql)
        number = cursor.fetchone()[-1]
        connect.commit()
        cursor.close()
        connect.close()
        return number

    # 用于遍历的查询，返回所有数据
    def iter_search(self):
        sql = '''
            SELECT * FROM face_data_set
        '''
        connect, cursor = self.connect_db()
        cursor.execute(sql)
        data = cursor.fetchall()
        connect.commit()
        cursor.close()
        connect.close()
        return data

    # 训练之后修改是否修改的字段
    def change_train_field(self):
        sql = '''
        UPDATE face_data_set
        SET FACE_IS_TRAINED = 1
        '''
        connect, cursor = self.connect_db()
        cursor.execute(sql)
        connect.commit()
        cursor.close()
        connect.close()
