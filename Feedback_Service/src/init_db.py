import pymysql

# Root credentials (must already exist)
root_user = "root"
root_password = "your_root_password"

# New DB and user details
db_name = "feedbackdb"
new_user = "feedback_user"
new_password = "strongpassword"

def init_db():
    try:
        # Connect as root
        connection = pymysql.connect(
            host="localhost",
            user=root_user,
            password=root_password
        )
        cursor = connection.cursor()

        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name};")

        # Create user
        cursor.execute(f"CREATE USER IF NOT EXISTS '{new_user}'@'localhost' IDENTIFIED BY '{new_password}';")

        # Grant permissions
        cursor.execute(f"GRANT ALL PRIVILEGES ON {db_name}.* TO '{new_user}'@'localhost';")
        cursor.execute("FLUSH PRIVILEGES;")

        print(f"✅ Database `{db_name}` and user `{new_user}` created successfully.")
    except Exception as e:
        print("❌ Error:", e)
    finally:
        connection.close()

if __name__ == "__main__":
    init_db()
