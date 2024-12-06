import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Định nghĩa hàm mục tiêu 3 tham số (x, y, z)
def ham_muc_tieu(x):
    return  x[0]**2 + x[1]**2 + x[2]**2 - 10 * np.cos(x[0]) - 10 * np.cos(x[1]) - 10 * np.cos(x[2])

# Khởi tạo tham số của PSO
n_particles = 20      # Số lượng hạt
n_iterations = 100    # Số vòng lặp
w = 0.7               # Hệ số quán tính
c1 = 1.4              # Hệ số học hỏi của hạt
c2 = 1.4              # Hệ số học hỏi của nhóm
d = 3                 # Số chiều (số tham số cần tối ưu, ví dụ: x, y, z)
x_min, x_max = -10, 10  # Khoảng tìm kiếm cho mỗi tham số (biên dưới, biên trên)
v_min, v_max = -2, 2  # Giới hạn vận tốc cho từng tham số

# Khởi tạo vị trí và tốc độ ngẫu nhiên
x = np.random.uniform(x_min, x_max, (n_particles, d))  # Vị trí ngẫu nhiên (mảng n_particles x d)
v = np.random.uniform(v_min, v_max, (n_particles, d))  # Vận tốc ngẫu nhiên (mảng n_particles x d)

# Khởi tạo giá trị tốt nhất cá nhân và toàn cục
p_best = x.copy()  # Mỗi hạt có giá trị tốt nhất ban đầu là vị trí của chính nó
f_best = np.apply_along_axis(ham_muc_tieu, 1, p_best)  # Mảng lưu giá trị hàm mục tiêu tại các vị trí đó
g_best = p_best[np.argmin(f_best)]  # Vị trí tốt nhất toàn cục
f_gbest = np.min(f_best)  # Giá trị hàm mục tiêu tại vị trí tốt nhất toàn cục

# Lưu các vị trí của các hạt qua các vòng lặp để vẽ đồ thị động
history = []

# Hàm cập nhật vị trí các hạt trong mỗi vòng lặp
def update_particles():
    global x, v, p_best, f_best, g_best, f_gbest,f_x
    
    r1 = np.random.rand(n_particles, d)
    r2 = np.random.rand(n_particles, d)
    
    # Cập nhật vận tốc và vị trí
    v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
    x = x + v
    
    # Giới hạn vị trí các hạt trong khoảng [x_min, x_max]
    x = np.clip(x, x_min, x_max)
    
    # Cập nhật giá trị tốt nhất cá nhân
    f_x = np.apply_along_axis(ham_muc_tieu, 1, x)
    better_mask = f_x < f_best
    p_best[better_mask] = x[better_mask]
    f_best[better_mask] = f_x[better_mask]
    
    # Cập nhật giá trị tốt nhất toàn cục
    if np.min(f_best) < f_gbest:
        f_gbest = np.min(f_best)
        g_best = p_best[np.argmin(f_best)]
    
    # Lưu lại các vị trí của các hạt cho từng vòng lặp
    history.append(x.copy())

# Tạo figure cho đồ thị 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Hàm vẽ đồ thị
def animate(i):
    ax.clear()  # Xóa đồ thị cũ
    # tạo 3 trục đồ thị 
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_zlim(x_min, x_max)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    # Vẽ các hạt
    x_data = history[i][:, 0]
    y_data = history[i][:, 1]
    z_data = history[i][:, 2]
    ax.scatter(x_data, y_data, z_data, c='r', marker='o')

    # Vẽ điểm tốt nhất toàn cục
    ax.scatter(g_best[0], g_best[1], g_best[2], c='b', marker='*', s=100)  # Vị trí tốt nhất toàn cục

# Cập nhật hạt qua các vòng lặp
for i in range(n_iterations):
    update_particles()
    if i % 10 == 0:
        print(f"Vòng lặp thứ {i}: f_gbest = {round(f_gbest, 3)}")
        for i in range(n_particles):
            print(f"Hạt {i + 1}: Vị trí = {np.round(x[i], 3)}, Vận tốc = {np.round(v[i], 3)}, f(x) = {np.round(f_x[i], 3)}")
   

# Tạo animation
ani = animation.FuncAnimation(fig, animate, frames=n_iterations, interval=200, repeat=True)
# Kết quả cuối cùng
print(f"Vị trí tốt nhất cuối cùng: {np.round(g_best, 3)}")
print(f"Giá trị hàm mục tiêu nhỏ nhất f(x): {round(f_gbest, 3)}")
# Hiển thị đồ thị
plt.show()

