import re
import bcrypt
from flask import redirect, url_for, render_template, request, session, flash
from db import get_db_connection

def signup(request):
    if request.method == 'POST':
        username = request.form['username']
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        try:
            supabase = get_db_connection()
            
            # Check if email already exists
            response = supabase.table('users').select("*").eq('email', email).execute()
            if response.data:
                flash("Email already registered, please log in.", "error")
                return redirect(url_for('signup_page'))

            # Check if first user (will be admin)
            response = supabase.table('users').select("*").execute()
            role = 'admin' if not response.data else 'user'

            # Insert new user
            response = supabase.table('users').insert({
                "username": username,
                "full_name": full_name,
                "email": email,
                "password": hashed_password.decode('utf-8'),
                "role": role
            }).execute()

            # Print user data for verification
            if response.data:
                print("New user created:")
                print(f"User ID: {response.data[0]['id']}")
                print(f"Username: {response.data[0]['username']}")
                print(f"Role: {response.data[0]['role']}")

            flash("You have successfully registered!", "success")
            return redirect(url_for('signin_page'))

        except Exception as e:
            print(f"Error: {e}")
            flash("Internal server error", "error")
            return redirect(url_for('signup_page'))

    return render_template('signup.html')

def signin(request):
    if request.method == 'POST':
        email_or_username = request.form['logemail']
        password = request.form['logpass']

        if re.match(r"[^@]+@[^@]+\.[^@]+", email_or_username):
            identifier = "email"
        else:
            identifier = "username"

        try:
            supabase = get_db_connection()

            # Query user by email or username
            if identifier == "email":
                response = supabase.table('users').select("*").eq('email', email_or_username).execute()
            else:
                response = supabase.table('users').select("*").eq('username', email_or_username).execute()

            if response.data and len(response.data) > 0:
                user = response.data[0]
                print(f"User found: ID={user['id']}, Role={user['role']}")  # Print for verification
                if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                    session['user'] = email_or_username
                    session['role'] = user['role']
                    session['user_id'] = user['id']  # Store user ID in session
                    
                    if user['role'] == "admin":
                        return redirect(url_for('admin_dashboard'))
                    else:
                        return redirect(url_for('user_dashboard'))
            
            flash("Invalid email/username or password", "error")
            return redirect(url_for('signin_page'))

        except Exception as e:
            print(f"Error: {e}")
            flash("Internal server error. Please try again later.", "error")
            return redirect(url_for('signin_page'))

    return render_template('signin.html')

def logout():
    # Clear all session data
    session.clear()
    flash("You have been logged out successfully!", "success")
    return redirect(url_for('signin_page'))
