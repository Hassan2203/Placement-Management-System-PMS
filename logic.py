"""
NAMAL PLACEMENT MANAGEMENT SYSTEM
Core Logic Module - PBL COMPLIANT with All 4 Custom Data Structures

DOMAIN: University Placement Management System
- Module A (Priority Processor): Min-Heap for CGPA-based student ranking
- Module B (Rapid Registry): Hash Table for O(1) student/opportunity lookup
- Module C (Connectivity Network): Weighted Graph + Dijkstra for city distances
- Module D (Organized Archive): BST for maintaining sorted student records
"""

import csv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

EMAIL_CONFIG = {
    'enabled': True, 
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'replace your mail', 
    'sender_password': 'two step varification password',
    'sender_name': 'Placement Office'
}

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_path(filename):
    return os.path.join(DATA_DIR, filename)

# ==================== MODULE A: PRIORITY PROCESSOR (MIN-HEAP) ====================
# CLO-2: Custom Min-Heap implementation for CGPA-based priority processing

class MinHeap:
    """
    Custom Min-Heap implementation (NO heapq library).
    Priority: Lower CGPA = Higher priority for academic support programs.
    """
    
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, student_data):
        """Insert student with CGPA priority - O(log n)"""
        self.heap.append(student_data)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        """Maintain min-heap property upward"""
        while i > 0:
            parent_idx = self.parent(i)
            # Ensure both CGPAs are float for comparison
            current_cgpa = float(self.heap[i]['cgpa']) if isinstance(self.heap[i]['cgpa'], str) else self.heap[i]['cgpa']
            parent_cgpa = float(self.heap[parent_idx]['cgpa']) if isinstance(self.heap[parent_idx]['cgpa'], str) else self.heap[parent_idx]['cgpa']
            
            if current_cgpa < parent_cgpa:
                self.swap(i, parent_idx)
                i = parent_idx
            else:
                break
    
    def extract_min(self):
        """Extract student with lowest CGPA - O(log n)"""
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_student = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return min_student
    
    def _heapify_down(self, i):
        """Maintain min-heap property downward"""
        while True:
            smallest = i
            left = self.left_child(i)
            right = self.right_child(i)
            
            # Ensure CGPAs are float for comparison
            smallest_cgpa = float(self.heap[smallest]['cgpa']) if isinstance(self.heap[smallest]['cgpa'], str) else self.heap[smallest]['cgpa']
            
            if left < len(self.heap):
                left_cgpa = float(self.heap[left]['cgpa']) if isinstance(self.heap[left]['cgpa'], str) else self.heap[left]['cgpa']
                if left_cgpa < smallest_cgpa:
                    smallest = left
                    smallest_cgpa = left_cgpa
            
            if right < len(self.heap):
                right_cgpa = float(self.heap[right]['cgpa']) if isinstance(self.heap[right]['cgpa'], str) else self.heap[right]['cgpa']
                if right_cgpa < smallest_cgpa:
                    smallest = right
            
            if smallest != i:
                self.swap(i, smallest)
                i = smallest
            else:
                break
    
    def peek(self):
        """View minimum without removing - O(1)"""
        return self.heap[0] if self.heap else None
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0

# ==================== MODULE B: RAPID REGISTRY (HASH TABLE) ====================
# CLO-1: Custom Hash Table with Chaining (NO dict for core structure)

class HashTable:
    """
    Custom Hash Table with Chaining collision resolution.
    Used for O(1) average-case student/opportunity lookup.
    """
    
    def __init__(self, size=100):
        self.size = size
        self.table = [[] for _ in range(size)]  # Chaining: list of lists
        self.count = 0
    
    def _hash(self, key):
        """Hash function: sum of ASCII values mod table size"""
        return sum(ord(char) for char in str(key)) % self.size
    
    def insert(self, key, value):
        """Insert key-value pair - O(1) average case"""
        index = self._hash(key)
        
        # Update if key exists
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        
        # Insert new key-value
        self.table[index].append((key, value))
        self.count += 1
    
    def search(self, key):
        """O(1) average case lookup"""
        index = self._hash(key)
        
        for k, v in self.table[index]:
            if k == key:
                return v
        
        return None
    
    def delete(self, key):
        """Remove key-value pair - O(1) average case"""
        index = self._hash(key)
        
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index].pop(i)
                self.count -= 1
                return True
        
        return False
    
    def get_all_values(self):
        """Return all stored values"""
        values = []
        for chain in self.table:
            for _, value in chain:
                values.append(value)
        return values
    
    def load_factor(self):
        """Calculate load factor α = n/m for complexity analysis"""
        return self.count / self.size

# ==================== MODULE C: CONNECTIVITY NETWORK (WEIGHTED GRAPH + DIJKSTRA) ====================
# CLO-3: Custom Graph with Dijkstra's algorithm (NO networkx)

class WeightedGraph:
    """
    Custom Weighted Graph with Dijkstra's shortest path algorithm.
    Represents Pakistani cities connected by roads with distance costs.
    """
    
    def __init__(self):
        self.adjacency_list = {}
    
    def add_vertex(self, vertex):
        """Add a city/location to graph"""
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
    
    def add_edge(self, vertex1, vertex2, weight):
        """Add bidirectional road between cities with distance"""
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        
        # Bidirectional (undirected graph)
        self.adjacency_list[vertex1].append((vertex2, weight))
        self.adjacency_list[vertex2].append((vertex1, weight))
    
    def dijkstra(self, start, end):
        """
        Dijkstra's algorithm for shortest path - O((V + E) log V)
        V = vertices (cities), E = edges (roads)
        """
        if start not in self.adjacency_list or end not in self.adjacency_list:
            return float('inf')
        
        if start == end:
            return 0
        
        # Initialize distances
        distances = {vertex: float('inf') for vertex in self.adjacency_list}
        distances[start] = 0
        
        # Priority queue: (distance, vertex)
        pq = [(0, start)]
        visited = set()
        
        while pq:
            # Manual min extraction (simulating priority queue without heapq)
            pq.sort()
            current_dist, current = pq.pop(0)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Found destination
            if current == end:
                return current_dist
            
            # Explore neighbors
            for neighbor, weight in self.adjacency_list[current]:
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    pq.append((distance, neighbor))
        
        return float('inf')

# ==================== MODULE D: ORGANIZED ARCHIVE (BINARY SEARCH TREE) ====================
# CLO-3: Custom BST for sorted student records

class BSTNode:
    """Node for Binary Search Tree"""
    def __init__(self, student_data):
        self.data = student_data
        self.left = None
        self.right = None

class BinarySearchTree:
    """
    Custom BST for maintaining students sorted by CGPA.
    Supports in-order traversal for sorted output.
    """
    
    def __init__(self):
        self.root = None
    
    def insert(self, student_data):
        """Insert student maintaining BST property - O(log n) average"""
        if self.root is None:
            self.root = BSTNode(student_data)
        else:
            self._insert_recursive(self.root, student_data)
    
    def _insert_recursive(self, node, student_data):
        """Recursive insertion"""
        # Ensure both CGPAs are float for comparison
        student_cgpa = float(student_data['cgpa']) if isinstance(student_data['cgpa'], str) else student_data['cgpa']
        node_cgpa = float(node.data['cgpa']) if isinstance(node.data['cgpa'], str) else node.data['cgpa']
        
        if student_cgpa < node_cgpa:
            if node.left is None:
                node.left = BSTNode(student_data)
            else:
                self._insert_recursive(node.left, student_data)
        else:
            if node.right is None:
                node.right = BSTNode(student_data)
            else:
                self._insert_recursive(node.right, student_data)
    
    def inorder_traversal(self):
        """Return students in sorted order (ascending CGPA) - O(n)"""
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        """Recursive in-order traversal"""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)
    
    def get_descending(self):
        """Return students sorted by CGPA (descending - highest first)"""
        return list(reversed(self.inorder_traversal()))

# ==================== GLOBAL DATA STRUCTURE INSTANCES ====================

# Module A: Priority queue for students needing support
student_priority_heap = MinHeap()

# Module B: Hash tables for instant lookup
student_registry = HashTable(size=100)
opportunity_registry = HashTable(size=50)
user_registry = HashTable(size=100)

# Module C: City connectivity graph
city_graph = WeightedGraph()

# Module D: BST for sorted student records
student_bst = BinarySearchTree()

# ==================== INITIALIZE GRAPH WITH CITIES ====================

def initialize_city_graph():
    """Initialize weighted graph with Pakistani cities and roads"""
    global city_graph
    
    edges = [
        ('Mianwali', 'Bhakkar', 70),
        ('Mianwali', 'Sargodha', 90),
        ('Mianwali', 'Chakwal', 130),
        ('Bhakkar', 'Faisalabad', 150),
        ('Sargodha', 'Faisalabad', 85),
        ('Sargodha', 'Lahore', 180),
        ('Chakwal', 'Talagang', 45),
        ('Chakwal', 'Islamabad', 90),
        ('Talagang', 'Islamabad', 110),
        ('Islamabad', 'Rawalpindi', 15),
        ('Islamabad', 'Peshawar', 170),
        ('Rawalpindi', 'Lahore', 350),
        ('Lahore', 'Faisalabad', 120),
        ('Lahore', 'Multan', 340),
        ('Faisalabad', 'Multan', 210),
        ('Multan', 'Hyderabad', 520),
        ('Karachi', 'Hyderabad', 165),
    ]
    
    for v1, v2, weight in edges:
        city_graph.add_edge(v1, v2, weight)

initialize_city_graph()

# ==================== CSV OPERATIONS ====================

def read_csv(filename):
    """Read CSV file and return list of dictionaries"""
    path = get_path(filename)
    data = []
    if not os.path.exists(path):
        return []
    try:
        with open(path, mode='r', encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row and any(row.values()):
                    clean_row = {}
                    for k, v in row.items():
                        if k and k.strip():
                            clean_row[k.strip()] = v.strip() if v else ''
                    if clean_row:
                        data.append(clean_row)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
    return data

def write_csv(filename, fieldnames, data):
    """Write list of dictionaries to CSV file"""
    path = get_path(filename)
    try:
        with open(path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"❌ Error writing {filename}: {e}")
        return False

def append_csv(filename, fieldnames, row_dict):
    """Append single row to CSV file"""
    path = get_path(filename)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    try:
        with open(path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)
        return True
    except Exception as e:
        print(f"❌ Error appending to {filename}: {e}")
        return False

# ==================== EMAIL FUNCTIONS ====================

def send_email(recipient_email, subject, body_html):
    """Send HTML email to recipient"""
    if not EMAIL_CONFIG.get('enabled', True):
        print("⚠️ Email sending is disabled")
        return False
    
    try:
        print(f"📧 Sending email to: {recipient_email}")
        message = MIMEMultipart('alternative')
        message['From'] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
        message['To'] = recipient_email
        message['Subject'] = subject
        
        html_part = MIMEText(body_html, 'html')
        message.attach(html_part)
        
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.send_message(message)
        server.quit()
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR sending email to {recipient_email}: {e}")
        return False

def send_welcome_email(student_name, student_email, reg_no, program, password):
    """Send welcome email with login credentials"""
    
    print(f"\n{'='*60}")
    print(f"📧 SENDING WELCOME EMAIL TO: {student_name}")
    print(f"   Email: {student_email}")
    print(f"   Username: {reg_no}")
    print(f"   Password: {password}")
    print(f"{'='*60}\n")
    
    subject = "🎓 Welcome to Namal Placement Portal - Your Login Credentials"
    
    body_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f4f4f4; }}
.email-container {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
.header {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 30px; text-align: center; }}
.header h1 {{ margin: 0; font-size: 24px; }}
.content {{ padding: 30px; }}
.credentials-box {{ background: #f8f9fa; padding: 25px; border-radius: 8px; margin: 25px 0; border-left: 4px solid #11998e; }}
.credential-item {{ padding: 10px 0; border-bottom: 1px solid #e9ecef; }}
.credential-label {{ font-weight: 600; color: #666; display: inline-block; width: 120px; }}
.credential-value {{ color: #333; font-weight: 700; font-size: 16px; }}
.password-highlight {{ background: #fff3cd; padding: 5px 10px; border-radius: 4px; color: #856404; font-family: 'Courier New', monospace; }}
.btn {{ display: inline-block; padding: 15px 40px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin: 20px 0; }}
.footer {{ background: #f8f9fa; padding: 20px; text-align: center; font-size: 14px; color: #6c757d; }}
</style>
</head>
<body>
<div class="email-container">
    <div class="header"><h1>🎓 Welcome to Namal Placement Portal</h1></div>
    <div class="content">
        <p style="font-size: 18px; color: #11998e; font-weight: 600;">Hello {student_name}! 👋</p>
        <p>Your account has been successfully created in the Namal University Placement Management System.</p>
        <div class="credentials-box">
            <h3 style="margin-top: 0; color: #11998e;">🔐 Your Login Credentials</h3>
            <div class="credential-item"><span class="credential-label">Username:</span><span class="credential-value">{reg_no}</span></div>
            <div class="credential-item"><span class="credential-label">Password:</span><span class="credential-value password-highlight">{password}</span></div>
            <div class="credential-item"><span class="credential-label">Program:</span><span class="credential-value">{program}</span></div>
            <div class="credential-item" style="border-bottom:none;"><span class="credential-label">Email:</span><span class="credential-value">{student_email}</span></div>
        </div>
        <center><a href="http://172.16.14.144:5000/login" class="btn">🚀 Login to Portal</a></center>
        <p><strong>⚠️ Important:</strong> Please keep your login credentials secure.</p>
    </div>
    <div class="footer"><p><strong>Namal University Placement Office</strong></p></div>
</div>
</body>
</html>"""
    
    result = send_email(student_email, subject, body_html)
    return result

def send_opportunity_notification(student_name, student_email, opportunity):
    """Send email notification for new opportunity"""
    
    print(f"📧 Sending opportunity notification to {student_name}")
    
    subject = f"🎯 New {opportunity['type']}: {opportunity['title']}"
    
    body_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; text-align: center; }}
.content {{ background: #fff; padding: 30px; border: 1px solid #e0e0e0; }}
.opportunity-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #28a745; }}
.btn {{ display: inline-block; padding: 12px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; margin: 20px 0; }}
.footer {{ background: #f8f9fa; padding: 20px; border-radius: 0 0 10px 10px; text-align: center; font-size: 14px; color: #6c757d; }}
</style>
</head>
<body>
<div class="header"><h1>🎯 New Opportunity Alert</h1></div>
<div class="content">
    <h2>Hello {student_name}!</h2>
    <p>A new <strong>{opportunity['type']}</strong> opportunity has been posted:</p>
    <div class="opportunity-box">
        <h3 style="margin-top:0;color:#28a745">💼 {opportunity['title']}</h3>
        <p><strong>Type:</strong> {opportunity['type']}<br>
        <strong>Location:</strong> 📍 {opportunity['location']}<br>
        <strong>Distance:</strong> 🚗 {opportunity.get('distance', 'N/A')} km<br>
        <strong>Min CGPA Required:</strong> {opportunity['min_cgpa']}</p>
    </div>
    <center><a href="{opportunity['link']}" class="btn">Apply Now →</a></center>
</div>
<div class="footer"><p><strong>Namal University Placement Office</strong></p></div>
</body>
</html>"""
    
    return send_email(student_email, subject, body_html)

# ==================== DATA SYNCHRONIZATION ====================

def sync_data_structures_from_csv():
    """
    Load CSV data into all four custom data structures.
    Called on startup and after updates.
    """
    global student_priority_heap, student_registry, opportunity_registry, user_registry, student_bst
    
    # Clear existing structures
    student_priority_heap = MinHeap()
    student_registry = HashTable(size=100)
    opportunity_registry = HashTable(size=50)
    user_registry = HashTable(size=100)
    student_bst = BinarySearchTree()
    
    # Load students
    students = read_csv('students.csv')
    for student in students:
        try:
            # Ensure CGPA is float
            cgpa_value = student.get('cgpa', '0')
            if isinstance(cgpa_value, str):
                student['cgpa'] = float(cgpa_value) if cgpa_value else 0.0
            else:
                student['cgpa'] = float(cgpa_value)
        except:
            student['cgpa'] = 0.0
        
        # Ensure other fields have defaults
        student['semester'] = student.get('semester', '1')
        student['year'] = student.get('year', '1')
        student['email'] = student.get('email', '')
        student['gpa_history'] = student.get('gpa_history', '')
        
        # MODULE A: Add to priority heap
        student_priority_heap.insert(student.copy())
        
        # MODULE B: Add to hash table
        student_registry.insert(student['reg_no'], student)
        
        # MODULE D: Add to BST
        student_bst.insert(student.copy())
    
    # Load opportunities
    opportunities = read_csv('opportunities.csv')
    for opp in opportunities:
        opportunity_registry.insert(opp['id'], opp)
    
    # Load users
    users = read_csv('users.csv')
    for user in users:
        user_registry.insert(user['username'], user)
    
    print(f"✅ Synced data structures: {len(students)} students")

# Initialize on module load
sync_data_structures_from_csv()

# ==================== USER AUTHENTICATION (MODULE B) ====================

def authenticate_user(username, password):
    """
    MODULE B: O(1) user lookup using Hash Table
    """
    user = user_registry.search(username)
    if user and user.get('password') == password:
        return user
    return None

# ==================== STUDENT MANAGEMENT ====================

def register_new_student(reg_no, name, email, program, password):
    """
    Register new student - Updates ALL 4 data structures
    Uses Module B for duplicate checking
    """
    
    if not email or '@' not in email:
        return False, "Invalid email address"
    
    # MODULE B: Check duplicate (O(1))
    if student_registry.search(reg_no) is not None:
        return False, "Student already exists"
    
    # Create student record with float CGPA for data structures
    student_row = {
        'reg_no': reg_no,
        'name': name,
        'email': email,
        'program': program,
        'semester': '1',
        'year': '1',
        'cgpa': 0.0,  # Float for data structures
        'gpa_history': '',
        'registered_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Create CSV row with string CGPA for file storage
    csv_row = student_row.copy()
    csv_row['cgpa'] = '0.00'
    
    # Save to CSV with string CGPA
    success1 = append_csv('students.csv',
        ['reg_no', 'name', 'email', 'program', 'semester', 'year', 'cgpa', 'gpa_history', 'registered_date'],
        csv_row)
    
    # Create user account
    user_row = {
        'username': reg_no,
        'password': password,
        'role': 'Student',
        'created_at': datetime.now().strftime('%Y-%m-%d')
    }
    
    success2 = append_csv('users.csv',
        ['username', 'password', 'role', 'created_at'],
        user_row)
    
    if success1 and success2:
        # Update ALL data structures with float CGPA
        student_priority_heap.insert(student_row.copy())   # MODULE A
        student_registry.insert(reg_no, student_row)       # MODULE B
        student_bst.insert(student_row.copy())             # MODULE D
        user_registry.insert(reg_no, user_row)             # MODULE B
        
        # Send welcome email
        email_sent = send_welcome_email(name, email, reg_no, program, password)
        
        if email_sent:
            return True, f"Student registered! Welcome email sent to {email}"
        else:
            return True, "Student registered but email failed."
    
    return False, "Error saving student data"

def get_student_by_regno(reg_no):
    """
    MODULE B: O(1) student lookup using Hash Table
    """
    return student_registry.search(reg_no)

def get_students_sorted_by_cgpa():
    """
    MODULE D: Return students sorted by CGPA using BST (descending order)
    Returns dictionary objects for HTML template compatibility
    """
    sorted_students = student_bst.get_descending()
    
    # Format CGPA as string with 2 decimal places (handle both float and string)
    for student in sorted_students:
        try:
            if isinstance(student['cgpa'], str):
                student['cgpa'] = f"{float(student['cgpa']):.2f}"
            else:
                student['cgpa'] = f"{student['cgpa']:.2f}"
        except:
            student['cgpa'] = "0.00"
    
    return sorted_students

def update_student_gpa(reg_no, new_gpa):
    """Update student GPA and resync all data structures"""
    students = read_csv('students.csv')
    updated = False
    
    for student in students:
        if student.get('reg_no') == reg_no:
            # Update GPA history
            history = student.get('gpa_history', '')
            if history:
                history += '|' + new_gpa
            else:
                history = new_gpa
            student['gpa_history'] = history
            
            # Calculate CGPA
            gpas = [float(g) for g in history.split('|') if g]
            cgpa = sum(gpas) / len(gpas) if gpas else 0.0
            student['cgpa'] = f"{cgpa:.2f}"
            
            # Update semester/year
            current_sem = int(student.get('semester', 1))
            current_sem += 1
            student['semester'] = str(current_sem)
            student['year'] = str((current_sem + 1) // 2)
            
            updated = True
            break
    
    if updated:
        fieldnames = ['reg_no', 'name', 'email', 'program', 'semester', 'year', 'cgpa', 'gpa_history', 'registered_date']
        write_csv('students.csv', fieldnames, students)
        
        # Resynchronize ALL data structures
        sync_data_structures_from_csv()
        
        return True, f"GPA updated! New CGPA: {student['cgpa']}"
    
    return False, "Student not found"

# ==================== OPPORTUNITY MANAGEMENT ====================

def get_eligible_students(min_cgpa):
    """Get students eligible based on CGPA threshold"""
    try:
        min_cgpa_float = float(min_cgpa)
        all_students = student_registry.get_all_values()
        eligible = [s for s in all_students if s['cgpa'] >= min_cgpa_float]
        return eligible
    except:
        return []

def post_new_opportunity(opp_id, title, opp_type, min_cgpa, link, details, location):
    """
    Post new opportunity using Modules B & C
    - Module B: Check duplicates
    - Module C: Calculate distance via Dijkstra
    """
    
    # MODULE B: Check duplicate (O(1))
    if opportunity_registry.search(opp_id) is not None:
        return False, "Opportunity ID already exists"
    
    # MODULE C: Calculate distance using Dijkstra's algorithm
    distance = city_graph.dijkstra('Mianwali', location)
    
    if distance == float('inf'):
        distance_str = "Unknown"
    else:
        distance_str = str(int(distance))
    
    # Create opportunity
    opportunity = {
        'id': opp_id,
        'title': title,
        'type': opp_type,
        'min_cgpa': min_cgpa,
        'link': link,
        'details': details,
        'location': location,
        'distance': distance_str,
        'posted_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Save to CSV
    fieldnames = ['id', 'title', 'type', 'min_cgpa', 'link', 'details', 'location', 'distance', 'posted_date']
    success = append_csv('opportunities.csv', fieldnames, opportunity)
    
    if success:
        # MODULE B: Add to hash table
        opportunity_registry.insert(opp_id, opportunity)
        
        # Send notifications to eligible students
        eligible_students = get_eligible_students(min_cgpa)
        emails_sent = 0
        
        for student in eligible_students:
            student_email = student.get('email')
            if student_email:
                if send_opportunity_notification(student['name'], student_email, opportunity):
                    emails_sent += 1
        
        return True, f"Opportunity posted! {emails_sent}/{len(eligible_students)} emails sent."
    
    return False, "Error saving opportunity"

def dijkstra_shortest_path(start, end):
    """
    MODULE C: Wrapper for Dijkstra's algorithm
    """
    return city_graph.dijkstra(start, end)

# ==================== STATISTICS ====================

def get_system_statistics():
    """Get system statistics"""
    all_students = student_registry.get_all_values()
    all_opportunities = opportunity_registry.get_all_values()
    
    total_students = len(all_students)
    active_students = len([s for s in all_students if s['cgpa'] > 0])
    
    cgpas = [s['cgpa'] for s in all_students if s.get('cgpa')]
    avg_cgpa = sum(cgpas) / len(cgpas) if cgpas else 0.0
    
    return {
        'total_students': total_students,
        'active_students': active_students,
        'total_opportunities': len(all_opportunities),
        'avg_cgpa': avg_cgpa
    }