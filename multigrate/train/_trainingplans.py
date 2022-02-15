from scvi.train import AdversarialTrainingPlan
from scvi import _CONSTANTS

class MILTrainingPlan(AdversarialTrainingPlan):

    def validation_step(self, batch, batch_idx):
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        reconstruction_loss = scvi_loss.reconstruction_loss
        self.log("validation_loss", scvi_loss.loss, on_epoch=True)
        return {
            "reconstruction_loss_sum": reconstruction_loss.sum(),
            "kl_local_sum": scvi_loss.kl_local.sum(),
            "kl_global": scvi_loss.kl_global,
            "n_obs": reconstruction_loss.shape[0],
            "integ_loss": scvi_loss.integ_loss,
            "cycle_loss": scvi_loss.cycle_loss,
            "class_loss": scvi_loss.class_loss,
            "accuracy": scvi_loss.accuracy,
            "reg_loss": scvi_loss.reg_loss,
            "regression_loss": scvi_loss.regression_loss,
        }

    def validation_epoch_end(self, outputs):
        """Aggregate validation step information."""
        n_obs, elbo, rec_loss, kl_local, integ, cycle, cl, acc, reg, regr = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            kl_local += tensors["kl_local_sum"]
            n_obs += tensors["n_obs"]
            integ += tensors["integ_loss"]
            cycle += tensors["cycle_loss"]
            cl += tensors["class_loss"]
            acc += tensors["accuracy"]
            reg += tensors["reg_loss"]
            regr += tensors["regression_loss"]
        # kl global same for each minibatch
        kl_global = outputs[0]["kl_global"]
        elbo += kl_global
        self.log("elbo_validation", elbo / n_obs)
        self.log("reconstruction_loss_validation", rec_loss / n_obs)
        self.log("kl_local_validation", kl_local / n_obs)
        self.log("kl_global_validation", kl_global)
        self.log("integ_validation", integ / n_obs)
        self.log("cycle_validation", cycle / n_obs)
        self.log("classification_validation", cl / n_obs)
        self.log("accuracy_validation", acc / len(outputs))
        self.log("reg_loss_validation", reg / n_obs)
        self.log("regression_validation", regr / n_obs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        batch_tensor = batch[_CONSTANTS.BATCH_KEY]
        if optimizer_idx == 0:
            loss_kwargs = dict(kl_weight=self.kl_weight)
            inference_outputs, _, scvi_loss = self.forward(
                batch, loss_kwargs=loss_kwargs
            )

            loss = scvi_loss.loss
            # fool classifier if doing adversarial training
            if kappa > 0 and self.adversarial_classifier is not False:
                z = inference_outputs["z"]
                fool_loss = self.loss_adversarial_classifier(z, batch_tensor, False)
                loss += fool_loss * kappa

            reconstruction_loss = scvi_loss.reconstruction_loss
            self.log("train_loss", loss, on_epoch=True)
            return {
                "loss": loss,
                "reconstruction_loss_sum": reconstruction_loss.sum().detach(),
                "kl_local_sum": scvi_loss.kl_local.sum().detach(),
                "kl_global": scvi_loss.kl_global.detach(),
                "n_obs": reconstruction_loss.shape[0],
                "integ_loss": scvi_loss.integ_loss.detach(),
                "cycle_loss": scvi_loss.cycle_loss.detach(),
                "class_loss": scvi_loss.class_loss.detach(),
                "accuracy": scvi_loss.accuracy.detach(),
                "reg_loss": scvi_loss.reg_loss.detach(),
                "regression_loss": scvi_loss.regression_loss.detach(),
            }

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if optimizer_idx == 1:
            inference_inputs = self.module._get_inference_input(batch)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            loss = self.loss_adversarial_classifier(z.detach(), batch_tensor, True)
            loss *= kappa

            return loss

    def training_epoch_end(self, outputs):
        # only report from optimizer one loss signature
        if self.adversarial_classifier:
            self.training_epoch_end_mil(outputs[0])
        else:
            self.training_epoch_end_mil(outputs)

    def training_epoch_end_mil(self, outputs):
        n_obs, elbo, rec_loss, kl_local, integ, cycle, cl, acc, reg, regr = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            kl_local += tensors["kl_local_sum"]
            n_obs += tensors["n_obs"]
            integ += tensors["integ_loss"]
            cycle += tensors["cycle_loss"]
            cl += tensors["class_loss"]
            acc += tensors["accuracy"]
            reg += tensors["reg_loss"]
            regr += tensors["regression_loss"]
        # kl global same for each minibatch
        kl_global = outputs[0]["kl_global"]
        elbo += kl_global
        self.log("elbo_train", elbo / n_obs)
        self.log("reconstruction_loss_train", rec_loss / n_obs)
        self.log("kl_local_train", kl_local / n_obs)
        self.log("kl_global_train", kl_global)
        self.log("integ_train", integ / n_obs)
        self.log("cycle_train", cycle / n_obs)
        self.log("classification_train", cl / n_obs)
        self.log("accuracy_train", acc / len(outputs))
        self.log("reg_loss_train", reg / n_obs)
        self.log("regression_train", regr / n_obs)

class MultiVAETrainingPlan(AdversarialTrainingPlan):

    def validation_step(self, batch, batch_idx):
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        reconstruction_loss = scvi_loss.reconstruction_loss
        self.log("validation_loss", scvi_loss.loss, on_epoch=True)
        return {
            "reconstruction_loss_sum": reconstruction_loss.sum(),
            "kl_local_sum": scvi_loss.kl_local.sum(),
            "kl_global": scvi_loss.kl_global,
            "n_obs": reconstruction_loss.shape[0],
            "integ_loss": scvi_loss.integ_loss,
            "cycle_loss": scvi_loss.cycle_loss,
            # "class_loss": scvi_loss.class_loss,
            # "accuracy": scvi_loss.accuracy,
            # "reg_loss": scvi_loss.reg_loss,
            # "regression_loss": scvi_loss.regression_loss,
        }

    def validation_epoch_end(self, outputs):
        """Aggregate validation step information."""
        n_obs, elbo, rec_loss, kl_local, integ, cycle, cl, acc, reg, regr = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            kl_local += tensors["kl_local_sum"]
            n_obs += tensors["n_obs"]
            integ += tensors["integ_loss"]
            cycle += tensors["cycle_loss"]
            # cl += tensors["class_loss"]
            # acc += tensors["accuracy"]
            # reg += tensors["reg_loss"]
            # regr += tensors["regression_loss"]
        # kl global same for each minibatch
        kl_global = outputs[0]["kl_global"]
        elbo += kl_global
        self.log("elbo_validation", elbo / n_obs)
        self.log("reconstruction_loss_validation", rec_loss / n_obs)
        self.log("kl_local_validation", kl_local / n_obs)
        self.log("kl_global_validation", kl_global)
        self.log("integ_validation", integ / n_obs)
        self.log("cycle_validation", cycle / n_obs)
        # self.log("classification_validation", cl / n_obs)
        # self.log("accuracy_validation", acc / len(outputs))
        # self.log("reg_loss_validation", reg / n_obs)
        # self.log("regression_validation", regr / n_obs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        batch_tensor = batch[_CONSTANTS.BATCH_KEY]
        if optimizer_idx == 0:
            loss_kwargs = dict(kl_weight=self.kl_weight)
            inference_outputs, _, scvi_loss = self.forward(
                batch, loss_kwargs=loss_kwargs
            )

            loss = scvi_loss.loss
            # fool classifier if doing adversarial training
            if kappa > 0 and self.adversarial_classifier is not False:
                z = inference_outputs["z"]
                fool_loss = self.loss_adversarial_classifier(z, batch_tensor, False)
                loss += fool_loss * kappa

            reconstruction_loss = scvi_loss.reconstruction_loss
            self.log("train_loss", loss, on_epoch=True)
            return {
                "loss": loss,
                "reconstruction_loss_sum": reconstruction_loss.sum().detach(),
                "kl_local_sum": scvi_loss.kl_local.sum().detach(),
                "kl_global": scvi_loss.kl_global.detach(),
                "n_obs": reconstruction_loss.shape[0],
                "integ_loss": scvi_loss.integ_loss.detach(),
                "cycle_loss": scvi_loss.cycle_loss.detach(),
                # "class_loss": scvi_loss.class_loss.detach(),
                # "accuracy": scvi_loss.accuracy.detach(),
                # "reg_loss": scvi_loss.reg_loss.detach(),
                # "regression_loss": scvi_loss.regression_loss.detach(),
            }

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if optimizer_idx == 1:
            inference_inputs = self.module._get_inference_input(batch)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            loss = self.loss_adversarial_classifier(z.detach(), batch_tensor, True)
            loss *= kappa

            return loss

    def training_epoch_end(self, outputs):
        # only report from optimizer one loss signature
        if self.adversarial_classifier:
            self.training_epoch_end_mil(outputs[0])
        else:
            self.training_epoch_end_mil(outputs)

    def training_epoch_end_mil(self, outputs):
        n_obs, elbo, rec_loss, kl_local, integ, cycle, cl, acc, reg, regr = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            kl_local += tensors["kl_local_sum"]
            n_obs += tensors["n_obs"]
            integ += tensors["integ_loss"]
            cycle += tensors["cycle_loss"]
            # cl += tensors["class_loss"]
            # acc += tensors["accuracy"]
            # reg += tensors["reg_loss"]
            # regr += tensors["regression_loss"]
        # kl global same for each minibatch
        kl_global = outputs[0]["kl_global"]
        elbo += kl_global
        self.log("elbo_train", elbo / n_obs)
        self.log("reconstruction_loss_train", rec_loss / n_obs)
        self.log("kl_local_train", kl_local / n_obs)
        self.log("kl_global_train", kl_global)
        self.log("integ_train", integ / n_obs)
        self.log("cycle_train", cycle / n_obs)
        # self.log("classification_train", cl / n_obs)
        # self.log("accuracy_train", acc / len(outputs))
        # self.log("reg_loss_train", reg / n_obs)
        # self.log("regression_train", regr / n_obs)