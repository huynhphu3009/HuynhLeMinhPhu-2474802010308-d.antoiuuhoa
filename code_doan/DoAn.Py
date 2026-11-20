import numpy as np
import matplotlib.pyplot as plt


# 1. ĐỊNH NGHĨA HÀM MỤC TIÊU VÀ ĐẠO HÀM (NỀN TẢNG LAB 2)

def objective_function(x):
    """Hàm mục tiêu cần tìm cực tiểu: f(x) = x^2 + 3x"""
    return x**2 + 3*x

def compute_gradient(x):
    """Đạo hàm bậc nhất của hàm mục tiêu: f'(x) = 2x + 3"""
    return 2*x + 3

def gradient_descent(start_point, learning_rate=0.1, n_iter=100, tolerance=1e-6):
    """
    Thuật toán Gradient Descent (Mở rộng từ Bài thực hành 2)
    """
    x = start_point
    history = [x]
    losses = [objective_function(x)]
    
    for i in range(n_iter):
        grad = compute_gradient(x)
        new_x = x - learning_rate * grad
        
        history.append(new_x)
        losses.append(objective_function(new_x))
        
        # Điều kiện dừng khi đạt độ dung sai
        if abs(new_x - x) < tolerance:
            print(f"[Info] Hội tụ tại bước lặp thứ {i+1} với learning_rate = {learning_rate}")
            break
            
        x = new_x
        
    return x, history, losses

# 2. CHẠY THỰC NGHIỆM VÀ XUẤT KẾT QUẢ

if __name__ == "__main__":
    print("--- BẮT ĐẦU CHẠY THỰC NGHIỆM ĐỒ ÁN TỐI ƯU HÓA ---")
    
    start_point = 7.0
    n_iter = 50
    
    # Chạy với tham số chuẩn (Learning Rate = 0.1)
    optimal_x, history_default, losses_default = gradient_descent(
        start_point=start_point, learning_rate=0.1, n_iter=n_iter
    )
    print(f"Nghiệm tối ưu tìm được: x = {optimal_x:.6f}")
    print(f"Giá trị cực tiểu f(x)    : {objective_function(optimal_x):.6f}\n")

    # 3. PHẦN MỞ RỘNG: PHÂN TÍCH ĐỘ NHẠY (SENSITIVITY ANALYSIS)
    # Khảo sát các mức Learning Rate khác nhau để lấy điểm cao

    learning_rates = [0.01, 0.05, 0.1, 0.4]
    experiment_results = {}

    print("--- KẾT QUẢ KHẢO SÁT CÁC LEARNING RATE ---")
    for lr in learning_rates:
        final_x, history, losses = gradient_descent(
            start_point=start_point, learning_rate=lr, n_iter=n_iter
        )
        experiment_results[lr] = losses
        print(f"lr = {lr:4.2f} --> x = {final_x:9.6f} | Tổng số bước: {len(history)}")


    # 4. TRỰC QUAN HÓA VÀ VẼ BIỂU ĐỒ (DÙNG ĐỂ CHÈN VÀO BÁO CÁO)

    plt.figure(figsize=(14, 5))

    # Biểu đồ 1: Tiến trình dịch chuyển của Gradient Descent trên hàm số
    plt.subplot(1, 2, 1)
    x_vals = np.linspace(-8, 8, 500)
    y_vals = objective_function(x_vals)
    plt.plot(x_vals, y_vals, label="f(x) = x^2 + 3x", color='blue', linewidth=2)
    plt.scatter(
        history_default, 
        [objective_function(x) for x in history_default], 
        color='red', zorder=5, label='Đường đi Gradient Descent (lr=0.1)'
    )
    plt.title("Tiến trình hội tụ tối ưu hóa bậc nhất (Lab 2)")
    plt.xlabel("Biến x")
    plt.ylabel("Giá trị f(x)")
    plt.legend()
    plt.grid(True)

    # Biểu đồ 2: So sánh độ nhạy hội tụ theo các Learning Rate khác nhau
    plt.subplot(1, 2, 2)
    for lr, losses in experiment_results.items():
        plt.plot(losses, marker='o', markersize=3, label=f'Learning Rate = {lr}')
        
    plt.title("Phân tích độ nhạy (Sensitivity Analysis theo Learning Rate)")
    plt.xlabel("Số vòng lặp (Iterations)")
    plt.ylabel("Giá trị hàm mục tiêu Loss f(x)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    print("\n[Thành công] Đã chạy xong! Đang hiển thị biểu đồ kết quả...")
    plt.show()