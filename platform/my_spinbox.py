from PySide6.QtWidgets import QDoubleSpinBox, QToolTip
from PySide6.QtGui import QValidator
from PySide6.QtCore import QPoint, Qt

class FreeInputSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 强制开启鼠标追踪，确保点击事件更灵敏
        self.setFocusPolicy(Qt.StrongFocus)

    def validate(self, input_str, pos):
        # 移除用户可能输入的千分位符或空格
        clean_str = input_str.replace(',', '.')
        try:
            val = float(clean_str)
            # 如果数值已经在范围内，告诉 Qt 这是“完全合格”的 (Acceptable)
            if self.minimum() <= val <= self.maximum():
                return QValidator.Acceptable, input_str, pos
        except ValueError:
            pass
            
        # 如果数值不在范围内，或者是删空了，告诉 Qt 这是“中间状态” (Intermediate)
        # 这样 Qt 就不会在输入时拦截你，也不会在失去焦点时乱跳
        return QValidator.Intermediate, input_str, pos

    def fixup(self, input_str):
        # 只有在输入彻底无法解析（比如全是字母）时才强制修正
        try:
            val = float(input_str)
            if val < self.minimum():
                self.setValue(self.minimum())
            elif val > self.maximum():
                self.setValue(self.maximum())
        except ValueError:
            self.setValue(self.minimum())

    def focusInEvent(self, event):
        # 确保在控件正下方弹出提示
        # 使用 self.rect().bottomLeft() 确保坐标相对于控件自身，再转为全局
        global_pos = self.mapToGlobal(self.rect().bottomLeft())
        
        # 稍微向下偏移 5 像素，避免遮挡边框
        global_pos += QPoint(0, 5)
        
        range_text = f"请输入范围: {self.minimum()} ~ {self.maximum()}"
        
        # 这里的参数 0 表示该 ToolTip 属于 self，显示时间由系统决定
        QToolTip.showText(global_pos, range_text, self)
        
        super().focusInEvent(event)