import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import GestureCNN, train_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, save_path, patience=20):
    best_acc = 0.0
    no_improve_epochs = 0  # 记录连续没有改善的epoch数

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[Epoch:{epoch + 1}/{num_epochs}] Train Loss: {epoch_loss:.4f}")

        # 验证
        avg_test_loss, accuracy = evaluate(model, test_loader, criterion)

        # 如果更好，则保存模型并重置计数器
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path)
            print(f"Model saved at epoch {epoch + 1}. Best Acc = {best_acc:.2f}%")
            no_improve_epochs = 0  # 重置计数器
        else:
            no_improve_epochs += 1  # 增加计数器
            print(f"No improvement for {no_improve_epochs} epochs. Best Acc so far = {best_acc:.2f}%")

            # 检查是否应该早停
            if no_improve_epochs >= patience:
                print(f"Early stopping after {epoch + 1} epochs without improvement.")
                break

    # 加载最佳模型用于最终评估
    model.load_state_dict(torch.load(save_path))
    return model, best_acc


def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader.dataset)
    avg_acc = 100.0 * correct / total
    print(f"Val Loss: {avg_loss:.4f}, Val Accuracy: {avg_acc:.2f}%")
    return avg_loss, avg_acc


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    num_epochs = 400
    save_path = "best_model.pth"
    patience = 30  # 设置耐心值：连续30个epoch没有改善就早停

    model = GestureCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 训练并获取最佳模型
    final_model, best_accuracy = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        save_path=save_path,
        patience=patience
    )

    # 最后再评估一次
    print("\nFinal evaluation with best model:")
    evaluate(final_model, test_loader, criterion)