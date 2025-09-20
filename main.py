import pyodbc

# Kết nối bằng Windows Authentication
conn = pyodbc.connect(
    r"DRIVER={ODBC Driver 17 for SQL Server};"
    r"SERVER=KINAS2K4\SQLEXPRESS;"
    r"DATABASE=MiniRecommendation;"              # hoặc thay bằng tên DB của bạn
    r"Trusted_Connection=yes;"
)

cursor = conn.cursor()
cursor.execute("SELECT name FROM sys.databases;")

for row in cursor.fetchall():
    print(row)

cursor.close()
conn.close()
