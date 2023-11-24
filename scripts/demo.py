import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
import torch.utils.data as td
import torch_geometric.datasets as tgd
import torch_geometric.transforms as tgt
import tqdm

from eqx3d.pointnet import PointNet


def main():
    dataset = tgd.ShapeNet(
        root="/data/shapenet",
        include_normals=False,
        split="train",
        transform=tgt.Compose(
            [
                tgt.FixedPoints(num=1024, replace=False, allow_duplicates=True),
                lambda data: data.to_dict(),
                # Function which transposes the position matrix, but retains the full dict.
                # This is so it can be C x N, but the other keys are still N x C.
                lambda data: {k: v.T if k == "pos" else v for k, v in data.items()},
            ]
        ),
    )

    test_dataset = tgd.ShapeNet(
        root="/data/shapenet",
        include_normals=False,
        split="test",
        transform=tgt.Compose(
            [
                tgt.FixedPoints(num=1024, replace=False, allow_duplicates=True),
                lambda data: data.to_dict(),
                # Function which transposes the position matrix, but retains the full dict.
                # This is so it can be C x N, but the other keys are still N x C.
                lambda data: {k: v.T if k == "pos" else v for k, v in data.items()},
            ]
        ),
    )

    seed = 1234

    train_dataset, val_dataset = td.random_split(
        dataset,
        [0.8, 0.2],
        generator=torch.Generator().manual_seed(seed),
    )

    # Simple function to convert a batch from torch to jax.
    def collate_torch2jax(batch):
        batch = td.default_collate(batch)
        return jax.tree_map(lambda x: jnp.array(x), batch)

    BATCH_SIZE = 128

    train_loader = td.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_torch2jax
    )
    val_loader = td.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_torch2jax
    )
    test_loader = td.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_torch2jax
    )

    batch = next(iter(train_loader))

    # model = STN3d(3, jax.random.PRNGKey(0))
    # model = PointNetEncoder(3, jax.random.PRNGKey(0))
    init_model = PointNet(k=16, key=jax.random.PRNGKey(0))
    init_state = eqx.nn.State(init_model)

    keys = jax.random.split(jax.random.PRNGKey(1234), BATCH_SIZE)

    # inference_model = eqx.tree_inference(model, value=True)

    res, _ = jax.vmap(
        init_model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None)
    )(batch["pos"], init_state, keys)

    # res, _ = inference_model(batch["pos"][0].T, state)

    opt = optax.adam(1e-3)

    @eqx.filter_jit
    def compute_accuracy(model, state, batch, key):
        inference_model = eqx.tree_inference(model, True)

        # Make a prediction.
        keys = jax.random.split(key, batch["pos"].shape[0])
        pred_y, _ = jax.vmap(
            inference_model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None)
        )(batch["pos"], state, keys)

        pred_y = jnp.argmax(pred_y, axis=1)

        return jnp.mean(batch["category"][..., 0] == pred_y)

    def loss_fn(model, state, batch, key):
        keys = jax.random.split(key, batch["pos"].shape[0])
        logits, state = jax.vmap(
            model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None)
        )(batch["pos"], state, keys)

        # Negative log likelihood
        labels = jax.nn.one_hot(batch["category"][..., 0], 16)
        loss = -jnp.mean(jnp.sum(logits * labels, axis=-1))
        return loss, state

    def evaluate(model, state, loader, key):
        accuracy = 0.0
        n_batches = 0

        for batch in tqdm.tqdm(loader):
            _, key = jax.random.split(key)
            accuracy += compute_accuracy(model, state, batch, key)
            n_batches += 1

        accuracy /= n_batches

        return accuracy

    def train(
        model,
        train_loader,
        val_loader,
        opt: optax.GradientTransformation,
        key,
        epochs=100,
    ):
        # Initialize a state for the model.
        state = eqx.nn.State(model)

        # Initialize the optimizer.
        opt_state = opt.init(eqx.filter(model, eqx.is_array))

        # Define a train_step function.
        @eqx.filter_jit
        def train_step(model, state, opt_state, batch, key):
            # Compute the loss.
            (loss, state), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                model, state, batch, key
            )

            # Compute the gradients.
            updates, opt_state = opt.update(grad, opt_state, model)

            # Update the model.
            model = eqx.apply_updates(model, updates)

            return model, state, opt_state, loss

        # Run a single initial eval
        accuracy = evaluate(model, state, train_loader, key)
        print(f"Initial accuracy: {accuracy}")

        # Iterate over the data.
        for epoch in range(epochs):
            loss = 0.0
            n_batches = 0
            for batch in tqdm.tqdm(train_loader):
                _, key = jax.random.split(key)
                model, state, opt_state, loss = train_step(
                    model, state, opt_state, batch, key
                )

                loss += loss
                n_batches += 1

            loss /= n_batches

            print(f"Epoch {epoch}, loss: {loss}")

            if epoch % 5 == 0:
                # Evaluate the model.
                eval_accuracy = evaluate(model, state, val_loader, key)
                print(f"Epoch {epoch}, val accuracy: {eval_accuracy}")

        return model, state

    trained_model, trained_state = train(
        init_model, train_loader, val_loader, opt, jax.random.PRNGKey(0), epochs=1000
    )

    # loss = loss_fn(model, state, batch, keys)
    train_acc = evaluate(
        trained_model, trained_state, train_loader, jax.random.PRNGKey(0)
    )

    print(f"Train accuracy: {train_acc}")

    val_acc = evaluate(trained_model, trained_state, val_loader, jax.random.PRNGKey(0))

    print(f"Validation accuracy: {val_acc}")

    test_acc = evaluate(
        trained_model, trained_state, test_loader, jax.random.PRNGKey(0)
    )

    print(f"Test accuracy: {test_acc}")


if __name__ == "__main__":
    main()
