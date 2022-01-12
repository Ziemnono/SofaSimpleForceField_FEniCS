#pragma once

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/OptionsGroup.h>
#include <Eigen/Eigen>
#include <SofaCaribou/config.h>
#include <Caribou/Geometry/Tetrahedron.h>

#include <SofaCaribou/config.h>
#include <SofaCaribou/Topology/CaribouTopology.h>
#include <Caribou/config.h>
#include <Caribou/constants.h>
#include <Caribou/Geometry/Element.h>
#include <Caribou/Topology/Mesh.h>

using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::core::behavior;
using namespace sofa::core::topology;
using sofa::defaulttype::Vec3Types;

class SVKElasticForcefield_SOFA : public ForceField<Vec3Types> {
public:
    SOFA_CLASS(SVKElasticForcefield_SOFA, SOFA_TEMPLATE(ForceField, Vec3Types));

    // Type definitions
    using Tetrahedron = caribou::geometry::Tetrahedron<caribou::Linear>;
    using Inherit  = ForceField<Vec3Types>;
    using DataTypes = typename Inherit::DataTypes;
    using VecCoord = typename DataTypes::VecCoord;
    using VecDeriv = typename DataTypes::VecDeriv;
    using Coord    = typename DataTypes::Coord;
    using Deriv    = typename DataTypes::Deriv;
    using Real     = typename Coord::value_type;

    template<int nRows, int nColumns, int Options=Eigen::RowMajor>
    using Matrix = Eigen::Matrix<Real, nRows, nColumns, Options>;

    using Mat33   = Matrix<3, 3>;

    template<int nRows, int Options=0>
    using Vector = Eigen::Matrix<Real, nRows, 1, Options>;

    template<int nRows>
    using MapVector = Eigen::Map<const Vector<nRows>>;

    static constexpr INTEGER_TYPE Dimension = 3;
    static constexpr INTEGER_TYPE NumberOfNodes = Tetrahedron::NumberOfNodesAtCompileTime;
    static constexpr INTEGER_TYPE NumberOfNodesPerElement = caribou::geometry::traits<Tetrahedron>::NumberOfNodesAtCompileTime;
    static constexpr INTEGER_TYPE NumberOfGaussNode = Tetrahedron::NumberOfGaussNodesAtCompileTime;
    static constexpr INTEGER_TYPE NumberOfGaussNodesPerElement = caribou::geometry::traits<Tetrahedron>::NumberOfGaussNodesAtCompileTime;


    template <typename ObjectType>
    using Link = SingleLink<SVKElasticForcefield_SOFA, ObjectType, BaseLink::FLAG_STRONGLINK>;

    // Data structures

    struct GaussNode {
        Real weight;
        Real jacobian_determinant;
        Matrix<NumberOfNodesPerElement, Dimension> dN_dx;
    };

    // The container of Gauss points (for each elements) is an array if the number of integration
    // points per element is known at compile time, or a dynamic vector otherwise.
    using GaussContainer = typename std::conditional<NumberOfGaussNodesPerElement != caribou::Dynamic,
            std::array<GaussNode, static_cast<std::size_t>(NumberOfGaussNodesPerElement)>,
            std::vector<GaussNode>>::type;

    // public methods
    SVKElasticForcefield_SOFA();

    void init() override;
    void draw(const sofa::core::visual::VisualParams* vparams) override;

    void addForce(
            const MechanicalParams* mparams,
            Data<VecDeriv>& d_f,
            const Data<VecCoord>& d_x,
            const Data<VecDeriv>& d_v) override;

    void addDForce(
            const MechanicalParams* /*mparams*/,
            Data<VecDeriv>& /*d_df*/,
            const Data<VecDeriv>& /*d_dx*/) override;

    void addKToMatrix(
            sofa::defaulttype::BaseMatrix * /*matrix*/,
            SReal /*kFact*/,
            unsigned int & /*offset*/) override;

    SReal getPotentialEnergy(
            const MechanicalParams* /* mparams */,
            const Data<VecCoord>& /* d_x */) const override;

    void computeBBox(const sofa::core::ExecParams* params, bool onlyVisible) override;

    /** Get the set of Gauss integration nodes of an element */
    inline auto gauss_nodes_of(std::size_t element_id) const -> const auto & {
        return p_elements_quadrature_nodes[element_id];
    }

    /** Get the number of elements contained in this field **/
    [[nodiscard]] inline
    auto number_of_elements() const noexcept -> std::size_t {
        return (p_topology ? p_topology->number_of_elements() : 0);
    }

    [[nodiscard]] inline
    auto topology() const noexcept -> typename SofaCaribou::topology::CaribouTopology<Tetrahedron>::SPtr {
        return p_topology;
    }

    /**
     *  Assemble the stiffness matrix K.
     *
     *  Since the stiffness matrix is function of the position vector x, i.e. K(x), this method will
     *  use the mechanical state vector used in the last call to addForce (usually Position or FreePosition).
     *  If another state vector should be used as the x, use instead the update_stiffness(x) method.
     *
     *  A reference to the assembled stiffness matrix K as a column major sparse matrix can be later
     *  obtained using the method K().
     *
     */
    virtual void assemble_stiffness();

    /**
     *  Assemble the stiffness matrix K.
     *
     *  Since the stiffness matrix is function of the position vector x, i.e. K(x), this method will
     *  use the data vector x passed as parameter. If the
     *
     *  A reference to the assembled stiffness matrix K as a column major sparse matrix can be later
     *  obtained using the method K().
     *
     */
    virtual void assemble_stiffness(const sofa::core::objectmodel::Data<VecCoord> & x);

    /**
     *  Assemble the stiffness matrix K.
     *
     *  Since the stiffness matrix is function of the position vector x, i.e. K(x), this method will
     *  use the position vector x passed as a Eigen matrix nx3 parameter with n the number of nodes.
     *
     *  A reference to the assembled stiffness matrix K as a column major sparse matrix can be later
     *  obtained using the method K().
     *
     */
     template <typename Derived>
    void assemble_stiffness(const Eigen::MatrixBase<Derived> & x);

//private:
//    /** (Re)Compute the tangent stiffness matrix */
//    void compute_K();

//    template <typename T>
//    inline
//    Tetrahedron tetrahedron(std::size_t tetrahedron_id, const T & x) const
//    {
//        auto * topology = d_topology_container.get();
//        const auto &node_indices = topology->getTetrahedron(static_cast<sofa::Index>(tetrahedron_id));

//        Matrix<NumberOfNodes, 3> m;
//        for (Eigen::Index j = 0; j < NumberOfNodes; ++j) {
//            const auto &node_id = node_indices[static_cast<sofa::Index>(j)];
//            m.row(j) = MapVector<3>(&x[node_id][0]);
//        }

//        return Tetrahedron(m);
//    }

private:
    Data< Real > d_youngModulus;
    Data< Real > d_poissonRatio;
    Link<sofa::core::objectmodel::BaseObject> d_topology_container;

private:
//    std::vector<GaussNode> p_quadrature_nodes; // Linear tetrahedrons only have 1 gauss node per element
    std::vector<std::array<GaussNode, NumberOfGaussNode>> p_gauss_nodes; ///< Set of Gauss nodes per elements
    std::vector<GaussContainer> p_elements_quadrature_nodes;
    sofa::core::ConstMultiVecCoordId p_X_id = sofa::core::ConstVecCoordId::position();
    Eigen::SparseMatrix<Real> p_K;


private:
    /** Get the set of Gauss integration nodes of the given element */
    virtual auto get_gauss_nodes(const std::size_t & element_id, const Tetrahedron & element) const -> GaussContainer;
    void triangulate_face(const Tetrahedron & e, const std::size_t & face_id, std::vector<sofa::type::Vector3> & triangles_nodes);



// Private variables
/// Pointer to a CaribouTopology. This pointer will be null if a CaribouTopology
/// is found within the scene graph and linked using the d_topology_container data
/// parameter. Otherwise, if a compatible SOFA's topology (see SofaCaribou::topology::CaribouTopology::mesh_is_compatible())
/// is found and linked, an internal CaribouTopology component will be created
/// and its pointer will be stored here.
typename SofaCaribou::topology::CaribouTopology<Tetrahedron>::SPtr p_topology;

};
