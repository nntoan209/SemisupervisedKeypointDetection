import torch
import torch.nn as nn

class MeanTeacherNetwork(nn.Module):
    def __init__(self,
                 student_model,
                 teacher_model):
        super(MeanTeacherNetwork, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
            
        #  copy weight from student to teacher
        for t_param, s_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            t_param.data.copy_(s_param.data)  
            
        # turn off gradient for teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            param.detach_()
    
    def forward(self, inputs):
        # forward the teacher model
        with torch.no_grad():
            t_heatmap = self.teacher_model(inputs)
            
        # forward the student model
        s_heatmap  = self.student_model(inputs)
        
        return t_heatmap, s_heatmap
    
    def predict(self, items, cuda=True):
        with torch.no_grad():
            # predicted keypoints of the teacher model
            t_keypoints_pred, t_keypoints_pred_score = self.teacher_model.predict(items, cuda)
            
            # predicted keypoints of the student model
            s_keypoints_pred, s_keypoints_pred_score = self.student_model.predict(items, cuda)
            
        return t_keypoints_pred, t_keypoints_pred_score, s_keypoints_pred, s_keypoints_pred_score
    
    def predict_on_input_image(self, items, cuda=True):
        with torch.no_grad():
            # predicted keypoints of the teacher model
            t_keypoints_pred, t_keypoints_pred_score = self.teacher_model.predict_on_input_image(items, cuda)
            
            # predicted keypoints of the student model
            s_keypoints_pred, s_keypoints_pred_score = self.student_model.predict_on_input_image(items, cuda)
            
        return t_keypoints_pred, t_keypoints_pred_score, s_keypoints_pred, s_keypoints_pred_score

    def _update_teacher_ema(self, ema_decay):
        for t_param, s_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            t_param.data.mul_(ema_decay).add_(other=s_param.data, alpha=1 - ema_decay)
            