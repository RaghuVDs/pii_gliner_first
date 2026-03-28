import React from "react";
import { Navigate } from "react-router-dom";
import { Result } from "antd";
import { useAuthStore } from "@/store/authStore";
import { UserRole } from "@/types/enums";

const ROLE_HIERARCHY: Record<UserRole, number> = {
  [UserRole.SUPER_ADMIN]: 100,
  [UserRole.TENANT_ADMIN]: 80,
  [UserRole.ANALYST]: 60,
  [UserRole.API_USER]: 40,
  [UserRole.VIEWER]: 20,
};

interface RoleGuardProps {
  minRole: UserRole;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

const RoleGuard: React.FC<RoleGuardProps> = ({
  minRole,
  children,
  fallback,
}) => {
  const user = useAuthStore((state) => state.user);

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  const userLevel = ROLE_HIERARCHY[user.role] || 0;
  const requiredLevel = ROLE_HIERARCHY[minRole] || 0;

  if (userLevel < requiredLevel) {
    if (fallback) {
      return <>{fallback}</>;
    }
    return (
      <Result
        status="403"
        title="403"
        subTitle="Sorry, you do not have permission to access this page."
      />
    );
  }

  return <>{children}</>;
};

export default RoleGuard;
