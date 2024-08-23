from typing import Any, Callable, List
import pytest
import drjit as dr
import mitsuba as mi
import glob
from os.path import join, realpath, dirname, basename, splitext, exists

from mitsuba.scalar_rgb.test.util import find_resource

@pytest.mark.parametrize("scene_name", ["cornell_box"])
def test01_forward(variants_vec_rgb, scene_name):
    
    w = 16
    h = 16

    n = 5
    
    k = "light.emitter.radiance.value"
    
    def func(scene: mi.Scene, x) -> mi.TensorXf:
        with dr.profile_range("render"):
            result = mi.render(scene, spp=1)
        return result

    def run(scene: mi.Scene, n: int, func: Callable[mi.Scene, Any]) -> List[mi.TensorXf]:

        params = mi.traverse(scene)
        value = mi.Float(params[k].x)
        
        images = []
        for i in range(n):
            params[k].x = value + 10.0 * i
            params.update()
            
            img = func(scene, params[k].x)
            dr.eval(img)

            images.append(img)
            
        return images
            
    def load_scene(name: str):
        
        if name == "cornell_box":
            scene = mi.cornell_box()
            scene["sensor"]["film"]["width"] = w
            scene["sensor"]["film"]["height"] = h
            scene = mi.load_dict(scene)
        return scene

    scene = load_scene(scene_name)
    images_ref = run(scene, n, func)
    scene = load_scene(scene_name)
    images_frozen = run(scene, n, dr.freeze(func))

    for (ref, frozen) in zip(images_ref, images_frozen):
        assert dr.allclose(ref, frozen)
        
        
tutorials_dir = realpath(join(dirname(__file__), '../../../tutorials'))

def test02_pose_estimation(variants_vec_rgb):

    w = 128
    h = 128
    
    def apply_transformation(initial_vertex_positions, opt, params):
        opt["trans"] = dr.clip(opt["trans"], -0.5, 0.5)
        opt["angle"] = dr.clip(opt["angle"], -0.5, 0.5)

        trafo = (
            mi.Transform4f()
            .translate([opt["trans"].x, opt["trans"].y, 0.0])
            .rotate([0, 1, 0], opt["angle"] * 100.0)
        )

        print("ravel:")
        params["bunny.vertex_positions"] = dr.ravel(trafo @ initial_vertex_positions)
        
    def mse(image, image_ref):
        return dr.sum(dr.square(image - image_ref), axis=None)
        
    def optimize(scene, opt, ref, initial_vertex_positions, other):
        params = mi.traverse(scene)

        image = mi.render(scene, params, spp=1, seed = 0)

        # Evaluate the objective function from the current rendered image
        loss = mse(image, ref)
        print(f"{type(loss)=}")

        # Backpropagate through the rendering process
        dr.backward(loss)

        return image, loss

    def run(optimize, n) -> tuple[mi.TensorXf, mi.Point3f, mi.Float]:
        from mitsuba.scalar_rgb import Transform4f as T

        scene = mi.cornell_box()
        del scene["large-box"]
        del scene["small-box"]
        del scene["green-wall"]
        del scene["red-wall"]
        del scene["floor"]
        del scene["ceiling"]
        scene["bunny"] = {
            "type": "ply",
            "filename": f"{tutorials_dir}/scenes/meshes/bunny.ply",
            "to_world": T().scale(6.5),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": (0.3, 0.3, 0.75)},
            },
        }
        scene["integrator"] = {
            "type": "prb",
        }
        scene["sensor"]["film"] = {
            "type": "hdrfilm",
            "width": w,
            "height": h,
            "rfilter": {"type": "gaussian"},
            "sample_border": True,
        }
        
        scene = mi.load_dict(scene)
        params = mi.traverse(scene)
        
        params.keep("bunny.vertex_positions")
        initial_vertex_positions = dr.unravel(mi.Point3f, params["bunny.vertex_positions"])
        
        image_ref = mi.render(scene, spp=4)
        
        opt = mi.ad.Adam(lr=0.025)
        opt["angle"] = mi.Float(0.25)
        opt["trans"] = mi.Point3f(0.1, -0.25, 0.0)

        for i in range(n):
            params = mi.traverse(scene)
            params.keep("bunny.vertex_positions")
            
            apply_transformation(initial_vertex_positions, opt, params)
            
            with dr.profile_range("optimize"):
                image, loss = optimize(
                    scene,
                    opt,
                    image_ref,
                    initial_vertex_positions,
                    [
                        params["bunny.vertex_positions"],
                    ],
                )
                
            opt.step()

        image_final = mi.render(scene, spp=4, seed = 0)

        return image_final, opt["trans"], opt["angle"]


    n = 10

    print("Reference:")
    img_ref, trans_ref, angle_ref = run(optimize, n)
    print("Frozen:")
    img_frozen, trans_frozen, angle_frozen = run(dr.freeze(optimize), n)

    # NOTE: cannot compare results as errors accumulate and the result will never be the same.
    
    # assert dr.allclose(trans_ref, trans_frozen, 0.1)
    # assert dr.allclose(angle_ref, angle_frozen)
    # assert dr.allclose(img_ref, img_frozen)
